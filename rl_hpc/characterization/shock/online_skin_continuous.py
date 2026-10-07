#!/usr/bin/env python3
"""Continuous skin-only online FQI for shock/NEMD, with no time/step feature.

The action is an absolute real-valued skin in [0.25, 1.0]. Balance is always
applied with neighbour weight 1.5. The policy receives prior application
state, never decision number, timestep, shock position or co-runner label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import time

import numpy as np
import psutil

import collect_trajectories as collector
import cpu_jitter
import online_learn as old
import shock_counterfactual as base

SKIN_MIN, SKIN_MAX = .25, 1.0
FACTOR = 1.5
FEATURE_VERSION = 'shock-skin-continuous-no-progress-v1'
DATA_VERSION = 'shock-skin-continuous-v1'
GRID = np.linspace(SKIN_MIN, SKIN_MAX, 41)


def state_vector(state: dict) -> np.ndarray:
    """Only pre-decision application state; no step/phase/front proxy."""
    p = state['previous_segment']
    steps = max(1, p['steps'])
    timings = p['timing_avg_seconds']
    vertices = []
    for line in state['partition_mesh'].splitlines():
        fields = line.split()
        if len(fields) == 5:
            try:
                vertices.append(float(fields[2]))
            except ValueError:
                pass
    if len(vertices) != 256:
        raise RuntimeError('expected 32 brick domains in balance mesh')
    widths = [max(vertices[i:i + 8]) - min(vertices[i:i + 8])
              for i in range(0, 256, 8)]
    ratio = lambda key, scale: float(timings.get(key, 0.0)) / steps / scale
    return np.array([
        1.0,
        (state['temperature'] - 140.0) / 50.0,
        (state['pressure'] - 200.0) / 200.0,
        (state['atom_imbalance'] - 1.0) / 1.5,
        (state['skin'] - .5) / .5,
        ratio('pair', .02), ratio('neigh', .01), ratio('comm', .02),
        ratio('modify', .01), p['neighbor_builds'] / steps,
        (p['nlocal']['max'] - p['nlocal']['mean']) / max(1, p['nlocal']['mean']),
        (p['neighs']['max'] - p['neighs']['mean']) / max(1, p['neighs']['mean']),
        min(widths) / (old.LX / 32), max(widths) / (old.LX / 32),
        float(np.std(widths)) / (old.LX / 32),
    ], dtype=float)


def features(state: dict, skin: float) -> np.ndarray:
    s = np.clip(state_vector(state), -5, 5)
    a = (skin - .5) / .5
    delta = abs(skin - state['skin']) / .75
    changed = float(abs(skin - state['skin']) > 1e-8)
    return np.concatenate((s, [a, a * a, changed, delta], s[1:] * a))


class SkinFQI(old.OnlineFQI):
    def add_transition(self, row: dict) -> None:
        self.replay.append(row)
        self.phi.append(features(row['state'], row['skin']))
        candidates = np.concatenate((GRID, [row['next_state']['skin']]))
        self.next_candidates.append(np.stack([
            features(row['next_state'], float(skin)) for skin in candidates]))

    def q(self, state: dict, skin: float) -> float:
        if self.coef is None:
            return 0.0
        return float(np.clip(features(state, skin) @ self.coef, -50.0, 0.0))

    def choose(self, state: dict) -> float:
        """Maximize a piecewise quadratic Q exactly within each open side.

        The changed indicator makes 'stay' discontinuous; compare it
        explicitly. The finite-distance terms are quadratic on either side.
        """
        current = float(state['skin'])
        choices = [SKIN_MIN, SKIN_MAX, current]
        if self.coef is None:
            return current
        for lo, hi in ((SKIN_MIN, current), (current, SKIN_MAX)):
            if hi - lo < 1e-5:
                continue
            span = hi - lo
            xs = np.array([lo + .0001 * span, (lo + hi) / 2,
                           hi - .0001 * span])
            ys = np.array([self.q(state, float(x)) for x in xs])
            quadratic, linear, _constant = np.polyfit(xs, ys, 2)
            choices.extend(float(x) for x in xs)
            if quadratic < -1e-9:
                vertex = -linear / (2 * quadratic)
                if lo < vertex < hi:
                    choices.append(float(vertex))
        return max(choices, key=lambda skin: self.q(state, skin))


def exploratory_skin(rng: random.Random, decision: int, state: dict,
                     episode_seed: int) -> float:
    # 15 shuffled strata per block, plus a 20% explicit no-change option.
    if rng.random() < .20:
        return float(state['skin'])
    block = decision // 15
    position = decision % 15
    order = list(range(15))
    random.Random(0xABCDEF + block * 1009 + episode_seed).shuffle(order)
    bin_index = order[position]
    return SKIN_MIN + (bin_index + rng.random()) * (SKIN_MAX - SKIN_MIN) / 15


def action_commands(skin: float, previous_skin: float, decision: int) -> str:
    if not SKIN_MIN <= skin <= SKIN_MAX:
        raise RuntimeError('skin outside approved domain')
    lines = ['variable begin timer']
    if abs(skin - previous_skin) > 1e-8:
        lines.extend([f'neighbor {skin:.12g} bin',
                      'neigh_modify every 1 delay 0 check yes',
                      'run 0 post no'])
    lines.extend([
        'variable prepared timer',
        f'balance 1.0 shift x 10 1.0 weight neigh {FACTOR:g} '
        f'out partition-{decision:03d}.mesh',
        'variable balanced timer', 'run 500', 'variable ended timer',
        'print "TIMES $(v_ended-v_begin:%.9f) $(v_prepared-v_begin:%.9f) '
        '$(v_balanced-v_prepared:%.9f) $(v_ended-v_balanced:%.9f)"',
        collector.OBS,
    ])
    return '\n'.join(lines) + '\n'


def run_episode(out: Path, episode: int, args: argparse.Namespace,
                restart: Path, physical_seed: int, learner: SkinFQI,
                cores: tuple[int, ...] | None) -> dict:
    directory = out / f'episode-{episode:03d}'
    directory.mkdir()
    rng = random.Random(args.seed + episode)
    schedule = (cpu_jitter.schedule(args.decisions,
                (args.seed + episode) ^ 0x5EEDCAFE, args.jitter_min,
                args.jitter_max) if cores else [False] * args.decisions)
    (directory / 'contention_schedule.json').write_text(json.dumps({
        'active_by_decision': schedule, 'cores': list(cores or ()),
        'seed': (args.seed + episode) ^ 0x5EEDCAFE,
        'min_duration': args.jitter_min, 'max_duration': args.jitter_max},
        indent=2) + '\n')
    contender = cpu_jitter.CpuJitter(cores) if cores else None
    sim = None
    status = 'failed'
    start = time.monotonic()
    rewards = []
    try:
        if contender:
            contender.start()
            contender.set_active(schedule[0])
        sim = collector.Simulation(directory)
        if episode == args.explore_episodes and learner.coef is None and learner.replay:
            update = learner.fit()
            with (directory / 'model_updates.jsonl').open('a') as stream:
                stream.write(json.dumps(update) + '\n')
        initial = rng.uniform(SKIN_MIN, SKIN_MAX)
        setup = base.fork_input(restart, initial, 'none', 500, 50).split(
            'variable action_start timer')[0]
        setup += (f'\nneighbor {initial:.12g} bin\nrun 0 post no\n'
                  f'balance 1.0 shift x 10 1.0 weight neigh {FACTOR:g} '
                  'out initial.mesh\nrun 50\n' + collector.OBS)
        state = collector.observation(sim.execute(setup), initial, FACTOR,
                                      expected_atoms=args.expected_atoms)
        state['partition_mesh'] = (directory / 'initial.mesh').read_text()
        with (directory / 'transitions.jsonl').open('w') as stream:
            for t in range(args.decisions):
                before = state
                if contender and contender.is_active != schedule[t]:
                    contender.set_active(schedule[t])
                    time.sleep(.15)
                phase = ('exploration' if episode < args.explore_episodes
                         else 'online_learning')
                epsilon = (1.0 if phase == 'exploration' else max(
                    .05, .25 - .20 * (episode - args.explore_episodes) /
                    max(1, args.episodes - args.explore_episodes - 1)))
                if phase == 'exploration' or learner.coef is None or rng.random() < epsilon:
                    skin = exploratory_skin(rng, t, before, args.seed + episode)
                    source = 'stratified' if phase == 'exploration' else 'epsilon'
                else:
                    skin = learner.choose(before)
                    source = 'greedy'
                # The policy sees only state_vector(before), not this label.
                worker_before = contender.cpu_seconds() if contender else 0.0
                cpu_before = (psutil.cpu_percent(interval=.10, percpu=True)
                              if contender else None)
                text = sim.execute(action_commands(skin, before['skin'], t))
                worker_after = contender.cpu_seconds() if contender else 0.0
                state = collector.observation(text, skin, FACTOR,
                                              expected_atoms=args.expected_atoms)
                if state['step'] != before['step'] + 500:
                    raise RuntimeError('non-500-step transition')
                if state['shock_position'] > .90 * old.LX:
                    raise RuntimeError('front-position guard exceeded')
                mesh = directory / f'partition-{t:03d}.mesh'
                state['partition_mesh'] = mesh.read_text() if mesh.exists() else before['partition_mesh']
                matches = re.findall(r'^TIMES (.*)$', text, re.M)
                if not matches:
                    raise RuntimeError('missing timer')
                times = list(map(float, matches[-1].split()))
                if (len(times) != 4 or not all(math.isfinite(x) and x >= 0 for x in times)
                        or abs(times[0] - sum(times[1:])) > 1e-6):
                    raise RuntimeError('invalid timer identity')
                reward = -times[0]
                row = {'version': DATA_VERSION, 'episode': episode, 'decision': t,
                       'timestamp': time.time(), 'physical_seed': physical_seed,
                       'phase': phase, 'jitter': 'cpu_piecewise' if cores else 'none',
                       'cpu_contention_active': schedule[t],
                       'cpu_contention_worker_seconds': max(0.0, worker_after - worker_before),
                       'cpu_utilization_target_cores_before':
                           ([cpu_before[core] for core in cores] if cores else None),
                       'state': before, 'action': {'skin': skin, 'factor': FACTOR},
                       'action_source': source, 'epsilon': epsilon,
                       'predicted_q': learner.q(before, skin) if learner.coef is not None else None,
                       'reward': reward, 'cumulative_reward': sum(rewards) + reward,
                       'prepare_seconds': times[1], 'balance_seconds': times[2],
                       'run_seconds': times[3], 'action_wall_seconds': times[0],
                       'next_state': state,
                       'neighbor_builds': state['previous_segment']['neighbor_builds'],
                       'dangerous_builds': state['previous_segment']['dangerous_builds'],
                       'safe': True}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                rewards.append(reward)
                learner.add_transition({'state': before, 'skin': skin, 'reward': reward,
                                        'next_state': state, 'decision': t})
                if phase == 'online_learning' and len(learner.replay) % args.update_every == 0:
                    update = learner.fit()
                    with (directory / 'model_updates.jsonl').open('a') as updates:
                        updates.write(json.dumps(update) + '\n')
                if t % 10 == 0 or t == args.decisions - 1:
                    print('episode', episode, 'decision', t, 'reward', round(reward, 3),
                          'cumulative', round(sum(rewards), 3), 'skin', round(skin, 5),
                          'source', source, 'updates', learner.updates, flush=True)
        code = sim.close(graceful=True)
        if code != 0:
            raise RuntimeError(f'MPI exit {code}')
        status = 'complete'
        result = {'episode': episode, 'decisions': args.decisions,
                  'physical_seed': physical_seed, 'restart': str(restart),
                  'end_step': state['step'], 'reward_total': sum(rewards),
                  'mean_reward': statistics.mean(rewards),
                  'wall_seconds': time.monotonic() - start,
                  'model_updates': learner.updates, 'process_exit': code}
        (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    except BaseException as exc:
        (directory / 'failure.json').write_text(json.dumps({
            'error': repr(exc), 'episode': episode, 'decision': locals().get('t'),
            'skin': locals().get('skin'), 'timestamp': time.time(),
            'recovery': 'episode excluded; no partial replay'}, indent=2) + '\n')
        raise
    finally:
        if sim is not None:
            sim.close()
        if contender is not None:
            contender.close()
        (directory / 'status.json').write_text(json.dumps(
            {'status': status, 'timestamp': time.time()}, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restart-list', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=36)
    parser.add_argument('--explore-episodes', type=int, default=12)
    parser.add_argument('--decisions', type=int, default=120)
    parser.add_argument('--expected-atoms', type=int, default=491520)
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--update-every', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=.95)
    parser.add_argument('--cpu-jitter', action='store_true')
    parser.add_argument('--cpu-cores', default='0-7')
    parser.add_argument('--jitter-min', type=int, default=2)
    parser.add_argument('--jitter-max', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.decisions <= 120 or not 0 <= args.explore_episodes < args.episodes:
        parser.error('invalid episode/decision counts')
    if not 0 <= args.gamma < 1 or args.update_every < 1:
        parser.error('invalid learner settings')
    if args.jitter_min < 1 or args.jitter_max < args.jitter_min:
        parser.error('invalid jitter duration')
    specs = json.loads(args.restart_list.read_text())
    if not specs:
        parser.error('empty restart list')
    restarts = [(Path(item['path']).resolve(), item['physical_seed']) for item in specs]
    if any(not path.is_file() for path, _ in restarts):
        parser.error('missing restart')
    cores = cpu_jitter.parse_cores(args.cpu_cores) if args.cpu_jitter else None
    out = args.output.resolve()
    if args.resume:
        if not (out / 'manifest.json').is_file():
            parser.error('resume needs existing manifest')
    else:
        out.mkdir(parents=True, exist_ok=False)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = {'version': DATA_VERSION, 'git_commit': base.git_commit(),
                'restarts': [{'path': str(path), 'physical_seed': seed,
                              'sha256': digest(path)} for path, seed in restarts],
                'binary': str(base.LMP), 'binary_sha256': digest(base.LMP),
                'script_sha256': digest(__file__), 'feature_version': FEATURE_VERSION,
                'excluded_policy_inputs': ['decision', 'step', 'shock_position',
                                           'CPU contention label', 'host load'],
                'skin_bounds': [SKIN_MIN, SKIN_MAX], 'balance_factor': FACTOR,
                'balance_each_decision': True, 'neighbor_policy': 'every 1 delay 0 check yes',
                'actions': 'absolute continuous real skin; unchanged skin explicitly allowed',
                'expected_atoms': args.expected_atoms, 'mpi_ranks': 32, 'omp_threads': 1,
                'steps_per_decision': 500, 'decisions': args.decisions,
                'episodes': args.episodes, 'explore_episodes': args.explore_episodes,
                'seed': args.seed, 'gamma': args.gamma, 'ridge': 100.0,
                'update_every': args.update_every,
                'jitter': 'cpu_piecewise' if cores else 'none',
                'cpu_cores': list(cores or ()), 'jitter_duration':
                    [args.jitter_min, args.jitter_max],
                'reward': 'negative action-application plus 500-step wall seconds',
                'physics': 'NVE and wall/reflect; diagnostic history omitted'}
    learner = SkinFQI(args.decisions, gamma=args.gamma, seed=args.seed)
    if args.resume:
        previous = json.loads((out / 'manifest.json').read_text())
        if previous != manifest:
            parser.error('resume code/configuration/restarts/binary differ')
        progress_file = out / 'progress.json'
        results = json.loads(progress_file.read_text())['results'] if progress_file.exists() else []
        for ep in range(len(results)):
            rows = [json.loads(line) for line in
                    (out / f'episode-{ep:03d}' / 'transitions.jsonl').read_text().splitlines()]
            if len(rows) != args.decisions:
                parser.error(f'incomplete committed episode {ep}')
            for row in rows:
                learner.add_transition({'state': row['state'],
                                        'skin': row['action']['skin'],
                                        'reward': row['reward'],
                                        'next_state': row['next_state'],
                                        'decision': row['decision']})
        if results:
            progress = json.loads(progress_file.read_text())
            learner.updates = progress['model_updates']
            learner.loss = progress['last_loss']
            policy = out / 'latest_policy.npz'
            if learner.updates:
                with np.load(policy, allow_pickle=False) as data:
                    if str(data['feature_version']) != FEATURE_VERSION:
                        parser.error('policy feature mismatch')
                    learner.coef = np.array(data['coef'], copy=True)
    else:
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        results = []
    for ep in range(len(results), args.episodes):
        attempt = out / f'episode-{ep:03d}'
        if attempt.exists():
            if not args.resume:
                raise RuntimeError('unexpected existing episode')
            preserved = out / f'interrupted-episode-{ep:03d}-{int(time.time())}'
            attempt.rename(preserved)
            print('PRESERVED', preserved, flush=True)
        restart, physical_seed = restarts[ep % len(restarts)]
        result = run_episode(out, ep, args, restart, physical_seed, learner, cores)
        results.append(result)
        if learner.coef is not None:
            np.savez(out / 'latest_policy.npz', coef=learner.coef,
                     feature_version=FEATURE_VERSION, gamma=args.gamma,
                     source_manifest=str(out / 'manifest.json'))
        (out / 'progress.json').write_text(json.dumps({
            'complete_episodes': len(results),
            'complete_transitions': sum(row['decisions'] for row in results),
            'results': results, 'model_updates': learner.updates,
            'last_loss': learner.loss}, indent=2) + '\n')
        print('COMPLETE episode', ep, 'reward', round(result['reward_total'], 3), flush=True)
    print('ALL COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(
        KeyboardInterrupt('skin study interrupted')))
    main()
