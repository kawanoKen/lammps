#!/usr/bin/env python3
"""Jitter-free online fitted-Q pilot for long shock/NEMD episodes.

The controller sees only prior-segment/application state. Each 500-step
transition is appended before any model update. This is an experimental
online learner, not evidence of a speedup without paired live evaluation.
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
import numeric_sweep as sweep
import shock_counterfactual as base

SKINS = tuple(sweep.SKINS)
FACTORS = tuple(sweep.FACTORS)
ACTIONS = [(skin, factor) for skin in SKINS for factor in FACTORS]
ACTIONS += [(skin, None) for skin in SKINS]  # keep current partition
OBS = collector.OBS
LX = 855.26277  # checked against the 491,520-atom nx=480 restart
FEATURE_VERSION = 'shock-online-linear-v2'


def action_dict(action):
    return {'skin': action[0], 'balance': action[1] is not None,
            'neighbor_weight_factor': action[1]}


def state_vector(state, decision, horizon):
    p = state['previous_segment']
    steps = max(1, p['steps'])
    timings = p['timing_avg_seconds']
    ratio = lambda key, scale: float(timings.get(key, 0.0)) / steps / scale
    vertices = []
    for line in state['partition_mesh'].splitlines():
        fields = line.split()
        if len(fields) == 5:
            try: vertices.append(float(fields[2]))
            except ValueError: pass
    if len(vertices) != 256:
        raise RuntimeError('expected 32 brick domains in balance mesh')
    widths = [max(vertices[i:i+8])-min(vertices[i:i+8])
              for i in range(0, 256, 8)]
    return np.array([
        1.0, decision / horizon, (decision / horizon)**2,
        state['shock_position'] / LX,
        (state['temperature'] - 140.0) / 50.0,
        (state['pressure'] - 200.0) / 200.0,
        (state['atom_imbalance'] - 1.0) / 1.5,
        (state['skin'] - 0.5) / 0.5,
        ((state['factor'] if state['factor'] is not None else 1.0) - 1.0) / 0.5,
        ratio('pair', .02), ratio('neigh', .01), ratio('comm', .02),
        ratio('modify', .01), p['neighbor_builds'] / steps,
        (p['nlocal']['max'] - p['nlocal']['mean']) / max(1, p['nlocal']['mean']),
        (p['neighs']['max'] - p['neighs']['mean']) / max(1, p['neighs']['mean']),
        min(widths) / (LX/32), max(widths) / (LX/32),
        float(np.std(widths)) / (LX/32),
    ], dtype=float)


def features(state, action, decision, horizon):
    s = np.clip(state_vector(state, decision, horizon), -5, 5)
    skin, factor = action
    a = np.array([(skin-.5)/.5,
                  (factor-1.0)/.5 if factor is not None else 0.0,
                  float(factor is None)], dtype=float)
    # Shared quadratic response surface plus state x action interactions.
    return np.concatenate((s, a, a*a, [a[0]*a[1]],
                           np.outer(s[1:], a).ravel()))


class OnlineFQI:
    def __init__(self, horizon, gamma=.95, ridge=100.0, seed=1):
        self.horizon, self.gamma, self.ridge = horizon, gamma, ridge
        self.rng = random.Random(seed)
        self.replay = []
        self.phi = []
        self.next_candidates = []
        self.coef = None
        self.loss = None
        self.updates = 0

    def values(self, state, decision):
        if self.coef is None:
            return np.zeros(len(ACTIONS))
        X = np.stack([features(state, action, decision, self.horizon)
                      for action in ACTIONS])
        return np.clip(X @ self.coef, -50.0, 0.0)

    def add_transition(self, row):
        self.replay.append(row)
        self.phi.append(features(row['state'], tuple(row['action']),
                                 row['decision'], self.horizon))
        self.next_candidates.append(np.stack([
            features(row['next_state'], action, row['decision']+1, self.horizon)
            for action in ACTIONS]))

    def fit(self):
        if not self.replay:
            return
        began = time.monotonic()
        X = np.stack(self.phi)
        next_X = np.stack(self.next_candidates)
        rewards = np.array([r['reward'] / 10.0 for r in self.replay])
        nonterminal = np.array([r['decision']+1 < self.horizon for r in self.replay])
        penalty = np.eye(X.shape[1]) * self.ridge
        penalty[0, 0] = 1e-6
        solver = np.linalg.solve(X.T @ X + penalty, X.T)
        w = solver @ rewards
        for _ in range(5 if self.gamma else 1):
            future = np.clip(next_X @ w, -50.0, 0.0).max(axis=1)
            future *= nonterminal
            targets = rewards + self.gamma * np.clip(future, -50.0, 0.0)
            w = solver @ targets
            if not np.isfinite(w).all():
                raise RuntimeError('nonfinite FQI coefficients')
        self.coef = w
        self.loss = float(np.mean((X @ w - targets)**2))
        self.updates += 1
        return dict(timestamp=time.time(), update=self.updates,
                    replay_size=len(self.replay), gamma=self.gamma,
                    ridge=self.ridge, train_td_mse=self.loss,
                    q_logged_min=float(np.min(X @ w)),
                    q_logged_max=float(np.max(X @ w)),
                    coef_l2=float(np.linalg.norm(w)),
                    update_wall_seconds=time.monotonic()-began)

def stratified_exploration(seed, horizon):
    rng = random.Random(seed)
    plan = []
    while len(plan) < horizon:
        block = list(ACTIONS)
        rng.shuffle(block)
        plan.extend(block)
    return plan[:horizon]


def commands(action, previous_skin, decision):
    skin, factor = action
    lines = ['variable begin timer']
    if skin != previous_skin:
        lines += [f'neighbor {skin:g} bin',
                  'neigh_modify every 1 delay 0 check yes', 'run 0 post no']
    lines.append('variable prepared timer')
    if factor is not None:
        # A current list is needed after any skin change; the initial setup
        # already builds a list before the first decision.
        lines.append(f'balance 1.0 shift x 10 1.0 weight neigh {factor:g} '
                     f'out partition-{decision:03d}.mesh')
    lines += ['variable balanced timer', 'run 500', 'variable ended timer',
              'print "TIMES $(v_ended-v_begin:%.9f) $(v_prepared-v_begin:%.9f) '
              '$(v_balanced-v_prepared:%.9f) $(v_ended-v_balanced:%.9f)"', OBS]
    return '\n'.join(lines) + '\n'


def run_episode(out, episode, horizon, explore_episodes, total_episodes, seed,
                restart, physical_seed,
                expected_atoms, learner, update_every,
                jitter_cores=None, jitter_min=2, jitter_max=5):
    directory = out / f'episode-{episode:03d}'
    directory.mkdir()
    sim = None
    contender = cpu_jitter.CpuJitter(jitter_cores) if jitter_cores else None
    contention_schedule = (cpu_jitter.schedule(horizon, seed ^ 0x5EEDCAFE,
                           jitter_min, jitter_max) if contender else [False] * horizon)
    (directory / 'contention_schedule.json').write_text(json.dumps(
        {'schedule_seed':seed ^ 0x5EEDCAFE, 'active_by_decision':contention_schedule,
         'cores':list(jitter_cores or []), 'min_duration':jitter_min,
         'max_duration':jitter_max}, indent=2)+'\n')
    rng = random.Random(seed)
    plan = stratified_exploration(seed, horizon)
    initial = rng.choice([(s, f) for s in SKINS for f in FACTORS])
    start = time.monotonic()
    status = 'failed'
    rewards = []
    try:
        if contender:
            contender.start()
            contender.set_active(contention_schedule[0])
        sim = collector.Simulation(directory)
        if episode == explore_episodes and learner.coef is None and learner.replay:
            update = learner.fit()
            with (directory / 'model_updates.jsonl').open('a') as updates:
                updates.write(json.dumps(update)+'\n')
        setup = base.fork_input(restart, initial[0], 'none', 500, 50).split(
            'variable action_start timer')[0]
        setup += (f'\nneighbor {initial[0]:g} bin\nrun 0 post no\n'
                  f'balance 1.0 shift x 10 1.0 weight neigh {initial[1]:g} '
                  'out initial.mesh\nrun 50\n' + OBS)
        state = collector.observation(sim.execute(setup), *initial,
                                      expected_atoms=expected_atoms)
        state['partition_mesh'] = (directory / 'initial.mesh').read_text()
        with (directory / 'transitions.jsonl').open('w') as stream:
            for t in range(horizon):
                before = state
                if contender and contender.is_active != contention_schedule[t]:
                    contender.set_active(contention_schedule[t])
                    time.sleep(.15)
                cpu_utilization_before = (psutil.cpu_percent(interval=.10, percpu=True)
                                          if contender else None)
                phase = 'exploration' if episode < explore_episodes else 'online_learning'
                epsilon = 1.0 if phase == 'exploration' else max(
                    .05, .25 - .20 * (episode-explore_episodes) /
                    max(1, total_episodes-explore_episodes-1))
                q_before = learner.values(before, t)
                if phase == 'exploration':
                    action, source = plan[t], 'stratified'
                elif learner.coef is None or rng.random() < epsilon:
                    action, source = rng.choice(ACTIONS), 'epsilon'
                else:
                    best = max(q_before)
                    picks = [i for i, q in enumerate(q_before) if q >= best-1e-10]
                    action, source = ACTIONS[rng.choice(picks)], 'greedy'
                if action not in ACTIONS:
                    raise RuntimeError('action outside approved grid')
                host_load_before = os.getloadavg()
                host_memory_before = psutil.virtual_memory().available
                contender_cpu_before = contender.cpu_seconds() if contender else 0.0
                begin_wall = time.time()
                begin_mono = time.monotonic()
                text = sim.execute(commands(action, before['skin'], t))
                end_mono = time.monotonic()
                end_wall = time.time()
                contender_cpu_after = contender.cpu_seconds() if contender else 0.0
                (directory / f'segment-{t:03d}.txt').write_text(text)
                factor_after = action[1] if action[1] is not None else before['factor']
                after = collector.observation(text, action[0], factor_after,
                                              expected_atoms=expected_atoms)
                if after['step'] != before['step'] + 500:
                    raise RuntimeError('non-500-step transition')
                if after['shock_position'] > .90 * LX:
                    raise RuntimeError('front-position guard exceeded')
                mesh = directory / f'partition-{t:03d}.mesh'
                after['partition_mesh'] = mesh.read_text() if mesh.exists() else before['partition_mesh']
                matches = re.findall(r'^TIMES (.*)$', text, re.M)
                if not matches:
                    raise RuntimeError('missing timer')
                times = list(map(float, matches[-1].split()))
                if (len(times) != 4 or not all(math.isfinite(x) and x >= 0 for x in times)
                        or abs(times[0]-sum(times[1:])) > 1e-6):
                    raise RuntimeError('invalid timer identity')
                reward = -times[0]
                row = dict(version='shock-online-v1', episode=episode, decision=t,
                           timestamp=time.time(), git_commit=base.git_commit(),
                           behavior_seed=seed, physical_seed=physical_seed,
                           restart=str(restart), phase=phase,
                           jitter='cpu_piecewise' if contender else 'none', mpi_ranks=32,
                           omp_threads=1, step_start=before['step'],
                           step_end=after['step'], state=before,
                           action=action_dict(action), action_index=ACTIONS.index(action),
                           action_source=source, epsilon=epsilon,
                           q_values_before=q_before.tolist(),
                           model_updates_before=learner.updates,
                           reward=reward, cumulative_reward=sum(rewards)+reward,
                           rolling_10_reward=statistics.mean((rewards+[reward])[-10:]),
                           action_start_timestamp=begin_wall,
                           action_end_timestamp=end_wall,
                           subprocess_roundtrip_seconds=end_mono-begin_mono,
                           host_load_average_before=host_load_before,
                           host_load_average_after=os.getloadavg(),
                           host_memory_available_before_bytes=host_memory_before,
                           cpu_contention_active=contention_schedule[t],
                           cpu_contention_cores=list(jitter_cores or []),
                           cpu_contention_worker_seconds=max(0.0,
                               contender_cpu_after-contender_cpu_before),
                           cpu_utilization_target_cores_before=(
                               [cpu_utilization_before[core] for core in jitter_cores]
                               if contender else None),
                           action_wall_seconds=times[0],
                           prepare_seconds=times[1], balance_seconds=times[2],
                           run_seconds=times[3], next_state=after,
                           neighbor_builds=after['previous_segment']['neighbor_builds'],
                           dangerous_builds=after['previous_segment']['dangerous_builds'],
                           safe=True, terminated=t == horizon-1)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
                rewards.append(reward)
                learner.add_transition(dict(state=before, action=action,
                                            reward=reward, next_state=after,
                                            decision=t))
                if phase == 'online_learning' and len(learner.replay) % update_every == 0:
                    update = learner.fit()
                    with (directory / 'model_updates.jsonl').open('a') as updates:
                        updates.write(json.dumps(update)+'\n')
                state = after
                if t % 10 == 0 or t == horizon-1:
                    print('episode', episode, 'decision', t, 'reward', round(reward, 3),
                          'cumulative', round(sum(rewards), 3), 'action', action,
                          'source', source, 'updates', learner.updates, flush=True)
        code = sim.close(graceful=True)
        if code != 0:
            raise RuntimeError(f'MPI exit {code}')
        status = 'complete'
        result = dict(episode=episode, decisions=horizon,
                      physical_seed=physical_seed, restart=str(restart),
                      start_step=state['step']-500*horizon,
                      end_step=state['step'], reward_total=sum(rewards),
                      mean_reward=statistics.mean(rewards),
                      wall_seconds=time.monotonic()-start,
                      model_updates=learner.updates, process_exit=code)
        (directory / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    except BaseException as exc:
        (directory / 'failure.json').write_text(json.dumps(
            {'error': repr(exc), 'time': time.time(), 'episode': episode,
             'decision': locals().get('t'),
             'attempted_action': action_dict(locals()['action'])
             if 'action' in locals() else None,
             'step_before_action': locals().get('before', {}).get('step')
             if isinstance(locals().get('before'), dict) else None,
             'recovery': 'episode terminated; unsafe/incomplete transition excluded from replay'},
            indent=2)+'\n')
        raise
    finally:
        if sim is not None:
            sim.close()
        if contender is not None:
            contender.close()
        (directory / 'status.json').write_text(json.dumps(
            {'status': status, 'time': time.time()}, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    restarts = p.add_mutually_exclusive_group(required=True)
    restarts.add_argument('--restart', type=Path)
    restarts.add_argument('--restart-list', type=Path,
                          help='JSON list of {path, physical_seed} records')
    p.add_argument('--episodes', type=int, default=36)
    p.add_argument('--explore-episodes', type=int, default=12)
    p.add_argument('--decisions', type=int, default=120)
    p.add_argument('--expected-atoms', type=int, default=491520)
    p.add_argument('--seed', type=int, default=20261002)
    p.add_argument('--update-every', type=int, default=20)
    p.add_argument('--gamma', type=float, default=.95)
    p.add_argument('--cpu-jitter', action='store_true',
                   help='toggle pinned CPU co-runner at decision boundaries')
    p.add_argument('--cpu-cores', default='0-7')
    p.add_argument('--jitter-min', type=int, default=2)
    p.add_argument('--jitter-max', type=int, default=5)
    p.add_argument('--resume', action='store_true',
                   help='resume after the last complete episode; preserve partial attempts')
    args = p.parse_args()
    def stop(_signum, _frame):
        raise KeyboardInterrupt('online run interrupted')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if not 1 <= args.decisions <= 120 or not 0 <= args.explore_episodes < args.episodes:
        p.error('decisions in [1,120], exploration episode count < total episodes')
    if not 0 <= args.gamma < 1 or args.update_every < 1:
        p.error('gamma in [0,1), update-every >= 1')
    if args.jitter_min < 1 or args.jitter_max < args.jitter_min:
        p.error('invalid jitter duration bounds')
    jitter_cores = cpu_jitter.parse_cores(args.cpu_cores) if args.cpu_jitter else None
    if args.restart_list:
        specs = json.loads(args.restart_list.read_text())
        if not isinstance(specs, list) or not specs:
            p.error('restart-list must be a nonempty JSON list')
        restart_specs = [(Path(item['path']).resolve(), item['physical_seed'])
                         for item in specs]
    else:
        restart_specs = [(args.restart.resolve(), 87287)]
    if any(not restart.is_file() for restart, _ in restart_specs):
        p.error('restart not found')
    out = args.output.resolve()
    if args.resume:
        if not (out / 'manifest.json').is_file():
            p.error('resume requires an existing manifest')
    else:
        out.mkdir(parents=True, exist_ok=False)
    manifest = dict(version='shock-online-v1', git_commit=base.git_commit(),
                    restarts=[dict(path=str(restart), physical_seed=seed,
                                   sha256=sweep.digest(restart))
                              for restart, seed in restart_specs],
                    binary=str(base.LMP), binary_sha256=sweep.digest(base.LMP),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    expected_atoms=args.expected_atoms, ranks=32, threads=1,
                    actions=[action_dict(a) for a in ACTIONS], steps_per_decision=500,
                    decisions=args.decisions, episodes=args.episodes,
                    explore_episodes=args.explore_episodes,
                    seed=args.seed, gamma=args.gamma, ridge=100.,
                    update_every=args.update_every, feature_version=FEATURE_VERSION,
                    jitter='cpu_piecewise' if jitter_cores else 'none',
                    cpu_jitter=(dict(cores=list(jitter_cores), min_duration=args.jitter_min,
                                     max_duration=args.jitter_max, idle_active_schedule='alternating')
                                if jitter_cores else None),
                    reward='negative action+run wall seconds',
                    physics='NVE and wall/reflect; CVCF/history diagnostics omitted')
    learner = OnlineFQI(args.decisions, gamma=args.gamma, seed=args.seed)
    results = []
    if args.resume:
        recorded = json.loads((out / 'manifest.json').read_text())
        if recorded != manifest:
            p.error('resume arguments, code, binary, or restart files differ from the manifest')
        progress_path = out / 'progress.json'
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            results = progress['results']
            for episode in range(len(results)):
                source = out / f'episode-{episode:03d}' / 'transitions.jsonl'
                rows = [json.loads(line) for line in source.read_text().splitlines()]
                if len(rows) != args.decisions:
                    p.error(f'incomplete committed episode {episode}')
                for row in rows:
                    action = row['action']
                    learner.add_transition(dict(
                        state=row['state'],
                        action=(action['skin'], action['neighbor_weight_factor']),
                        reward=row['reward'], next_state=row['next_state'],
                        decision=row['decision']))
            learner.updates = progress['model_updates']
            learner.loss = progress['last_loss']
            policy_path = out / 'latest_policy.npz'
            if learner.updates:
                if not policy_path.is_file():
                    p.error('saved model updates exist but latest_policy.npz is missing')
                with np.load(policy_path) as policy:
                    if str(policy['feature_version']) != FEATURE_VERSION:
                        p.error('policy feature version differs')
                    learner.coef = policy['coef']
        partial = out / f'episode-{len(results):03d}'
        if partial.exists():
            preserved = out / f'interrupted-episode-{len(results):03d}-{int(time.time())}'
            partial.rename(preserved)
            print('PRESERVED interrupted attempt', preserved, flush=True)
    else:
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for episode in range(len(results), args.episodes):
        restart, physical_seed = restart_specs[episode % len(restart_specs)]
        result = run_episode(out, episode, args.decisions,
                             args.explore_episodes, args.episodes,
                             args.seed+episode,
                             restart, physical_seed, args.expected_atoms, learner,
                             args.update_every, jitter_cores,
                             args.jitter_min, args.jitter_max)
        results.append(result)
        if learner.coef is not None:
            np.savez(out / 'latest_policy.npz', coef=learner.coef,
                     feature_version=FEATURE_VERSION, gamma=args.gamma,
                     source_manifest=str(out / 'manifest.json'))
        (out / 'progress.json').write_text(json.dumps(
            {'complete_episodes':len(results), 'complete_transitions':sum(
                r['decisions'] for r in results), 'results':results,
             'model_updates':learner.updates, 'last_loss':learner.loss}, indent=2)+'\n')
    print('COMPLETE', len(results), 'episodes', flush=True)


if __name__ == '__main__':
    main()
