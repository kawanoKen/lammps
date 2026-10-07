#!/usr/bin/env python3
"""Paired held-out live evaluation of frozen idle/CPU-jitter shock policies.

This is cross-regime testing, not K-fold retraining. Each policy gets its own
LAMMPS process, the same physical restart and the same contention schedule.
No model updates or exploration occur during evaluation.
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
import time

import numpy as np

import collect_trajectories as collector
import cpu_jitter
import online_learn as online
import shock_counterfactual as base


POLICIES = ('idle_trained', 'jitter_trained', 'fixed_05_15', 'fixed_035_15')
TRAINING = {
    'idle_trained': 'shock_online_idle_main_v1',
    'jitter_trained': 'shock_online_cpu_jitter_main_v1',
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_models() -> tuple[dict, dict]:
    root = base.ROOT / 'rl_hpc/characterization/data'
    models, provenance = {}, {}
    for name, directory in TRAINING.items():
        folder = root / directory
        source = folder / 'latest_policy.npz'
        manifest = json.loads((folder / 'manifest.json').read_text())
        progress = json.loads((folder / 'progress.json').read_text())
        if (progress['complete_episodes'] != manifest['episodes'] or
                manifest['actions'] != [online.action_dict(a) for a in online.ACTIONS] or
                manifest['feature_version'] != online.FEATURE_VERSION or
                manifest['decisions'] != 120 or
                manifest['binary_sha256'] != digest(base.LMP)):
            raise RuntimeError(f'incompatible or incomplete source model: {name}')
        with np.load(source, allow_pickle=False) as data:
            if str(data['feature_version']) != online.FEATURE_VERSION:
                raise RuntimeError(f'feature mismatch in {name}')
            coef = np.array(data['coef'], copy=True)
        if not np.isfinite(coef).all():
            raise RuntimeError(f'nonfinite model coefficients: {name}')
        models[name] = coef
        provenance[name] = {'dataset': directory, 'checkpoint_sha256': digest(source),
                            'manifest_sha256': digest(folder / 'manifest.json'),
                            'training_physical_seeds': [x['physical_seed']
                                                        for x in manifest['restarts']]}
    return models, provenance


def select(policy: str, state: dict, decision: int, models: dict) -> tuple:
    if policy == 'fixed_05_15':
        return (.5, 1.5)
    if policy == 'fixed_035_15':
        return (.35, 1.5)
    coef = models[policy]
    candidate = np.stack([online.features(state, a, decision, 120)
                          for a in online.ACTIONS])
    values = np.clip(candidate @ coef, -50.0, 0.0)
    return online.ACTIONS[int(np.argmax(values))]


def episode(root: Path, spec: dict, policy: str, regime: str, repeat: int,
            schedule: list[bool], models: dict, decisions: int,
            expected_atoms: int, cores: tuple[int, ...], selector=None) -> dict:
    identifier = f'seed-{spec["physical_seed"]}-{regime}-r{repeat}-{policy}'
    directory = root / identifier
    directory.mkdir()
    restart = Path(spec['path']).resolve()
    contender = cpu_jitter.CpuJitter(cores) if regime == 'jitter' else None
    sim = None
    status = 'failed'
    started = time.monotonic()
    initial = (.5, 1.5)
    rows = []
    try:
        if contender:
            contender.start()
            contender.set_active(schedule[0])
        sim = collector.Simulation(directory)
        setup = base.fork_input(restart, initial[0], 'none', 500, 50).split(
            'variable action_start timer')[0]
        setup += (f'\nneighbor {initial[0]:g} bin\nrun 0 post no\n'
                  f'balance 1.0 shift x 10 1.0 weight neigh {initial[1]:g} '
                  'out initial.mesh\nrun 50\n' + collector.OBS)
        state = collector.observation(sim.execute(setup), *initial,
                                      expected_atoms=expected_atoms)
        state['partition_mesh'] = (directory / 'initial.mesh').read_text()
        with (directory / 'transitions.jsonl').open('w') as stream:
            for t in range(decisions):
                if contender and contender.is_active != schedule[t]:
                    contender.set_active(schedule[t])
                    time.sleep(.15)
                action = (selector or select)(policy, state, t, models)
                before = state
                worker_before = contender.cpu_seconds() if contender else 0.0
                output = sim.execute(online.commands(action, state['skin'], t))
                worker_after = contender.cpu_seconds() if contender else 0.0
                next_factor = action[1] if action[1] is not None else state['factor']
                state = collector.observation(output, action[0], next_factor,
                                              expected_atoms=expected_atoms)
                mesh = directory / f'partition-{t:03d}.mesh'
                state['partition_mesh'] = (mesh.read_text() if mesh.exists()
                                           else before['partition_mesh'])
                matches = re.findall(r'^TIMES (.*)$', output, re.M)
                if not matches:
                    raise RuntimeError('missing timer output')
                times = [float(value) for value in matches[-1].split()]
                if (len(times) != 4 or not all(math.isfinite(v) and v >= 0 for v in times)
                        or abs(times[0] - sum(times[1:])) > 1e-6):
                    raise RuntimeError('invalid action timer')
                if state['step'] != before['step'] + 500:
                    raise RuntimeError('non-500-step transition')
                if state['shock_position'] > .90 * online.LX:
                    raise RuntimeError('front-position guard exceeded')
                dangerous = state['previous_segment']['dangerous_builds']
                if dangerous:
                    raise RuntimeError(f'dangerous neighbour builds: {dangerous}')
                row = {'decision': t, 'step_start': before['step'],
                       'step_end': state['step'], 'policy': policy,
                       'regime': regime, 'cpu_contention_active': schedule[t],
                       'action': online.action_dict(action), 'reward_seconds': -times[0],
                       'action_wall_seconds': times[0], 'prepare_seconds': times[1],
                       'balance_seconds': times[2], 'run_seconds': times[3],
                       'worker_cpu_seconds': max(0.0, worker_after - worker_before),
                       'temperature': state['temperature'],
                       'pressure': state['pressure'],
                       'energy_per_atom': state['total_energy_per_atom'],
                       'shock_position': state['shock_position'],
                       'atom_imbalance': state['atom_imbalance'],
                       'neighbor_builds': state['previous_segment']['neighbor_builds'],
                       'dangerous_builds': dangerous, 'timestamp': time.time()}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                rows.append(row)
                if t % 10 == 0 or t == decisions - 1:
                    print(identifier, 'decision', t, 'cumulative_s',
                          round(sum(x['action_wall_seconds'] for x in rows), 3),
                          flush=True)
        rc = sim.close(graceful=True)
        if rc != 0:
            raise RuntimeError(f'MPI exit status {rc}')
        status = 'complete'
        result = {'id': identifier, 'policy': policy, 'regime': regime,
                  'physical_seed': spec['physical_seed'], 'repeat': repeat,
                  'decisions': decisions, 'total_seconds': sum(
                      row['action_wall_seconds'] for row in rows),
                  'elapsed_seconds_including_startup': time.monotonic() - started,
                  'active_intervals': sum(schedule), 'end_step': state['step'],
                  'dangerous_builds': 0, 'exit_code': rc}
        (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    finally:
        if sim is not None:
            sim.close()
        if contender is not None:
            contender.close()
        (directory / 'status.json').write_text(json.dumps(
            {'status': status, 'timestamp': time.time(),
             'completed_decisions': len(rows)}, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restart-list', type=Path, required=True)
    parser.add_argument('--decisions', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--regimes', nargs='+', choices=('idle', 'jitter'),
                        default=['idle', 'jitter'])
    parser.add_argument('--policies', nargs='+', choices=POLICIES,
                        default=list(POLICIES))
    parser.add_argument('--expected-atoms', type=int, default=491520)
    parser.add_argument('--cpu-cores', default='0-7')
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--resume', action='store_true',
                        help='continue an interrupted run, preserving incomplete attempts')
    args = parser.parse_args()
    if not 1 <= args.decisions <= 120 or args.repeats < 1:
        parser.error('decisions must be 1..120 and repeats >= 1')
    if len(args.regimes) != len(set(args.regimes)) or len(args.policies) != len(set(args.policies)):
        parser.error('duplicate regime or policy')
    signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(
        KeyboardInterrupt('evaluation interrupted')))
    models, provenance = frozen_models()
    specs = json.loads(args.restart_list.read_text())
    if not specs:
        parser.error('empty restart list')
    train_seeds = {seed for item in provenance.values()
                   for seed in item['training_physical_seeds']}
    for spec in specs:
        path = Path(spec['path']).resolve()
        if spec['physical_seed'] in train_seeds:
            parser.error('restart was used in policy training')
        if not path.is_file() or digest(path) != spec['sha256']:
            parser.error(f'missing or changed restart: {path}')
    cores = cpu_jitter.parse_cores(args.cpu_cores) if 'jitter' in args.regimes else ()
    out = args.output.resolve()
    if args.resume:
        if not (out / 'manifest.json').is_file():
            parser.error('resume requires an existing manifest')
    else:
        out.mkdir(parents=True, exist_ok=False)
    schedules = {}
    jobs = []
    for repeat in range(args.repeats):
        for spec in specs:
            for regime in args.regimes:
                key = f'{spec["physical_seed"]}-{regime}-r{repeat}'
                schedule_seed = args.seed + 1009 * repeat + spec['physical_seed']
                schedule = (cpu_jitter.schedule(args.decisions, schedule_seed, 2, 5)
                            if regime == 'jitter' else [False] * args.decisions)
                schedules[key] = {'seed': schedule_seed, 'active_by_decision': schedule}
                policies = list(args.policies)
                random.Random(schedule_seed ^ 0xA11CE).shuffle(policies)
                jobs.extend((spec, regime, repeat, key, policy) for policy in policies)
    manifest = {'version': 'shock-online-heldout-cross-v1',
                'git_commit': base.git_commit(), 'lammps_binary': str(base.LMP),
                'lammps_sha256': digest(base.LMP), 'restart_list': str(args.restart_list.resolve()),
                'heldout_restarts': specs, 'frozen_policies': provenance,
                'policies': args.policies, 'regimes': args.regimes,
                'decisions': args.decisions, 'steps_per_decision': 500,
                'repeats': args.repeats, 'expected_atoms': args.expected_atoms,
                'mpi_ranks': 32, 'omp_threads': 1, 'cpu_cores': list(cores),
                'schedule': schedules, 'run_order': [
                    {'seed': spec['physical_seed'], 'regime': regime,
                     'repeat': repeat, 'policy': policy} for spec, regime, repeat, _, policy in jobs],
                'reward': 'negative action+run wall seconds; warmup/startup excluded',
                'evaluation': 'frozen, no online update, no exploration; one process per run',
                'script_sha256': digest(Path(__file__))}
    if args.resume:
        previous = json.loads((out / 'manifest.json').read_text())
        if previous != manifest:
            parser.error('resume configuration/code/model differs from the original manifest')
        results = (json.loads((out / 'results.json').read_text())
                   if (out / 'results.json').is_file() else [])
        expected_ids = [f'seed-{spec["physical_seed"]}-{regime}-r{repeat}-{policy}'
                        for spec, regime, repeat, _, policy in jobs]
        if [item['id'] for item in results] != expected_ids[:len(results)]:
            parser.error('completed result order does not match manifest')
    else:
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        results = []
    for spec, regime, repeat, key, policy in jobs[len(results):]:
        attempt = out / f'seed-{spec["physical_seed"]}-{regime}-r{repeat}-{policy}'
        if attempt.exists():
            if not args.resume:
                raise RuntimeError(f'unexpected existing output: {attempt}')
            preserved = attempt.with_name(attempt.name + f'-interrupted-{int(time.time())}')
            attempt.rename(preserved)
            print('PRESERVED', preserved, flush=True)
        result = episode(out, spec, policy, regime, repeat,
                         schedules[key]['active_by_decision'], models,
                         args.decisions, args.expected_atoms, cores)
        results.append(result)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('COMPLETE', result['id'], result['total_seconds'], flush=True)
    print('ALL COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    main()
