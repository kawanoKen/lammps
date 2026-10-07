#!/usr/bin/env python3
"""Paired live comparison of official fix-balance heuristics and learned policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import signal
import time

import numpy as np

import collect_trajectories as collector
import cpu_jitter
import online_balance_hybrid as hybrid
import shock_counterfactual as base


POLICIES = ('official_atoms', 'official_neigh_10', 'official_neigh_15',
            'official_time_10', 'learned_idle', 'learned_jitter')
TRAINING = {
    'learned_idle': 'shock_balance_hybrid_idle_v1',
    'learned_jitter': 'shock_balance_hybrid_cpu_jitter_v1',
}
NFREQ, THRESH, NITER, STOPTHRESH = 500, 1.2, 10, 1.1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models() -> tuple[dict, dict]:
    data = base.ROOT / 'rl_hpc/characterization/data'
    models, provenance = {}, {}
    for policy, dataset in TRAINING.items():
        folder = data / dataset
        manifest = json.loads((folder / 'manifest.json').read_text())
        progress = json.loads((folder / 'progress.json').read_text())
        checkpoint = folder / 'latest_policy.npz'
        if (progress['complete_episodes'] != manifest['episodes'] or
                manifest['feature_version'] != hybrid.FEATURE_VERSION or
                manifest['binary_sha256'] != digest(base.LMP)):
            raise RuntimeError(f'incompatible model: {policy}')
        learner = hybrid.BalanceFQI(manifest['decisions'],
                                    gamma=manifest['gamma'], seed=manifest['seed'])
        with np.load(checkpoint, allow_pickle=False) as saved:
            learner.coef = np.array(saved['coef'], copy=True)
        models[policy] = learner
        provenance[policy] = {
            'dataset': dataset, 'checkpoint_sha256': digest(checkpoint),
            'training_physical_seeds': [row['physical_seed']
                                        for row in manifest['restarts']],
        }
    return models, provenance


def fix_command(policy: str) -> str | None:
    prefix = (f'fix lb all balance {NFREQ} {THRESH} shift x '
              f'{NITER} {STOPTHRESH}')
    return {
        'official_atoms': prefix,
        'official_neigh_10': prefix + ' weight neigh 1.0',
        'official_neigh_15': prefix + ' weight neigh 1.5',
        'official_time_10': prefix + ' weight time 1.0',
    }.get(policy)


def official_commands() -> str:
    return '\n'.join([
        'variable begin timer', 'run 500', 'variable ended timer',
        'print "TIMES $(v_ended-v_begin:%.9f) 0.000000000 0.000000000 '
        '$(v_ended-v_begin:%.9f)"',
        'print "FIXBAL $(f_lb:%.9f) $(f_lb[1]:%.9f) $(f_lb[2]:%.9f) '
        '$(f_lb[3]:%.9f)"', collector.OBS,
    ]) + '\n'


def compact_state(state: dict) -> dict:
    """Return the observable state without the large partition-mesh payload."""
    return {key: value for key, value in state.items()
            if key != 'partition_mesh'}


def run_episode(root: Path, spec: dict, policy: str, regime: str, repeat: int,
                schedule: list[bool], models: dict, decisions: int,
                expected_atoms: int, cores: tuple[int, ...]) -> dict:
    identifier = f'seed-{spec["physical_seed"]}-{regime}-r{repeat}-{policy}'
    directory = root / identifier
    directory.mkdir()
    contender = cpu_jitter.CpuJitter(cores) if regime == 'jitter' else None
    sim, rows, status = None, [], 'failed'
    started = time.monotonic()
    try:
        if contender:
            contender.start()
            contender.set_active(schedule[0])
        sim = collector.Simulation(directory)
        restart = Path(spec['path']).resolve()
        setup = base.fork_input(restart, hybrid.SKIN, 'none', 500, 50).split(
            'variable action_start timer')[0]
        setup += (f'\nneighbor {hybrid.SKIN:g} bin\nrun 0 post no\n'
                  'balance 1.0 shift x 10 1.0 weight neigh 1.5 '
                  'out initial.mesh\nrun 50\n' + collector.OBS)
        state = collector.observation(sim.execute(setup), hybrid.SKIN, 1.5,
                                      expected_atoms=expected_atoms)
        state['partition_mesh'] = (directory / 'initial.mesh').read_text()
        hybrid.initialize_control_state(state)
        official = fix_command(policy)
        if official:
            sim.execute(official + '\n')
        with (directory / 'transitions.jsonl').open('w') as stream:
            for decision in range(decisions):
                if contender and contender.is_active != schedule[decision]:
                    contender.set_active(schedule[decision])
                    time.sleep(.15)
                before = state
                worker_before = contender.cpu_seconds() if contender else 0.0
                if official:
                    factor = None
                    output = sim.execute(official_commands())
                else:
                    factor = models[policy].choose(before)
                    output = sim.execute(hybrid.commands(factor, decision))
                worker_after = contender.cpu_seconds() if contender else 0.0
                next_factor = (before['factor'] if factor is None else factor)
                state = collector.observation(output, hybrid.SKIN, next_factor,
                                              expected_atoms=expected_atoms)
                mesh = directory / f'partition-{decision:03d}.mesh'
                state['partition_mesh'] = (mesh.read_text() if mesh.exists()
                                           else before['partition_mesh'])
                timer_matches = re.findall(r'^TIMES (.*)$', output, re.M)
                if not timer_matches:
                    raise RuntimeError('missing timer')
                times = [float(value) for value in timer_matches[-1].split()]
                if (len(times) != 4 or
                        not all(math.isfinite(value) and value >= 0 for value in times) or
                        abs(times[0] - sum(times[1:])) > 1e-6):
                    raise RuntimeError('invalid timer identity')
                if state['step'] != before['step'] + 500:
                    raise RuntimeError('non-500-step transition')
                if state['shock_position'] > .90 * hybrid.old.LX:
                    raise RuntimeError('front-position guard exceeded')
                dangerous = state['previous_segment']['dangerous_builds']
                if dangerous:
                    raise RuntimeError(f'dangerous neighbor builds: {dangerous}')
                fix_values = None
                if official:
                    fix_matches = re.findall(r'^FIXBAL (.*)$', output, re.M)
                    if not fix_matches:
                        raise RuntimeError('missing fix-balance telemetry')
                    fix_values = [float(value) for value in fix_matches[-1].split()]
                    if len(fix_values) != 4 or not all(map(math.isfinite, fix_values)):
                        raise RuntimeError('invalid fix-balance telemetry')
                    state.update(
                        intervals_since_balance=0,
                        previous_balance_applied=fix_values[2] > 0,
                        previous_balance_seconds=0.0,
                        previous_action_imbalance_before=fix_values[3],
                        previous_action_imbalance_after=fix_values[0])
                else:
                    state.update(
                        intervals_since_balance=(0 if factor is not None else
                                                 before['intervals_since_balance'] + 1),
                        previous_balance_applied=factor is not None,
                        previous_balance_seconds=times[2],
                        previous_action_imbalance_before=before['atom_imbalance'],
                        previous_action_imbalance_after=state['atom_imbalance'])
                row = {
                    'decision': decision, 'step_start': before['step'],
                    'step_end': state['step'], 'policy': policy, 'regime': regime,
                    'cpu_contention_active': schedule[decision],
                    'action': {'official_fix': official is not None,
                               'learned_factor': factor, 'skin': hybrid.SKIN},
                    'fix_balance': ({'imbalance_after': fix_values[0],
                                     'max_load': fix_values[1],
                                     'iterations': fix_values[2],
                                     'imbalance_before_last': fix_values[3]}
                                    if fix_values else None),
                    'action_wall_seconds': times[0], 'run_seconds': times[3],
                    'balance_seconds_explicit': times[2], 'reward': -times[0],
                    'worker_cpu_seconds': max(0.0, worker_after-worker_before),
                    'atom_imbalance': state['atom_imbalance'],
                    'temperature': state['temperature'], 'pressure': state['pressure'],
                    'energy_per_atom': state['total_energy_per_atom'],
                    'shock_position': state['shock_position'],
                    'neighbor_builds': state['previous_segment']['neighbor_builds'],
                    'dangerous_builds': dangerous, 'timestamp': time.time(),
                    'state_before': compact_state(before),
                    'state_after': compact_state(state),
                }
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                rows.append(row)
                if decision % 10 == 0 or decision == decisions-1:
                    print(identifier, decision,
                          round(sum(row['action_wall_seconds'] for row in rows), 3),
                          factor, flush=True)
        exit_code = sim.close(graceful=True)
        if exit_code:
            raise RuntimeError(f'MPI exit {exit_code}')
        status = 'complete'
        result = {
            'id': identifier, 'policy': policy, 'regime': regime,
            'physical_seed': spec['physical_seed'], 'repeat': repeat,
            'decisions': decisions,
            'total_seconds': sum(row['action_wall_seconds'] for row in rows),
            'elapsed_seconds_including_startup': time.monotonic()-started,
            'active_intervals': sum(schedule), 'end_step': state['step'],
            'dangerous_builds': 0, 'exit_code': exit_code,
        }
        (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    finally:
        if sim is not None:
            sim.close()
        if contender is not None:
            contender.close()
        (directory/'status.json').write_text(json.dumps({
            'status': status, 'timestamp': time.time(),
            'completed_decisions': len(rows)}, indent=2)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restart-list', type=Path, required=True)
    parser.add_argument('--decisions', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--regimes', nargs='+', choices=('idle','jitter'),
                        default=['idle','jitter'])
    parser.add_argument('--policies', nargs='+', choices=POLICIES,
                        default=list(POLICIES))
    parser.add_argument('--expected-atoms', type=int, default=491520)
    parser.add_argument('--cpu-cores', default='0-7')
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda _s,_f: (_ for _ in ()).throw(
        KeyboardInterrupt('evaluation interrupted')))
    models, provenance = load_models()
    specs = json.loads(args.restart_list.read_text())
    train_seeds = {seed for item in provenance.values()
                   for seed in item['training_physical_seeds']}
    for spec in specs:
        restart = Path(spec['path']).resolve()
        if spec['physical_seed'] in train_seeds:
            parser.error('held-out restart appeared in training')
        if not restart.is_file() or digest(restart) != spec['sha256']:
            parser.error(f'missing/changed restart: {restart}')
    cores = cpu_jitter.parse_cores(args.cpu_cores) if 'jitter' in args.regimes else ()
    output = args.output.resolve()
    if args.resume:
        if not (output/'manifest.json').is_file(): parser.error('resume requires manifest')
    else:
        output.mkdir(parents=True, exist_ok=False)
    schedules, jobs = {}, []
    for repeat in range(args.repeats):
        for spec in specs:
            for regime in args.regimes:
                key = f'{spec["physical_seed"]}-{regime}-r{repeat}'
                schedule_seed = args.seed + 1009*repeat + spec['physical_seed']
                schedule = (cpu_jitter.schedule(args.decisions, schedule_seed, 2, 5)
                            if regime == 'jitter' else [False]*args.decisions)
                schedules[key] = {'seed': schedule_seed, 'active_by_decision': schedule}
                order = list(args.policies)
                random.Random(schedule_seed ^ 0xF17BA1).shuffle(order)
                jobs.extend((spec,regime,repeat,key,policy) for policy in order)
    manifest = {
        'version':'shock-official-balance-heldout-v1',
        'git_commit':base.git_commit(), 'lammps_binary':str(base.LMP),
        'lammps_sha256':digest(base.LMP), 'heldout_restarts':specs,
        'frozen_policies':provenance, 'policies':args.policies,
        'regimes':args.regimes, 'decisions':args.decisions,
        'steps_per_decision':500, 'repeats':args.repeats,
        'expected_atoms':args.expected_atoms, 'mpi_ranks':32, 'omp_threads':1,
        'cpu_cores':list(cores), 'schedule':schedules,
        'official_parameters':{'Nfreq':NFREQ,'thresh':THRESH,'style':'shift x',
                               'Niter':NITER,'stopthresh':STOPTHRESH},
        'common_initial_balance':'weight neigh 1.5 followed by 50 warmup steps',
        'run_order':[{'seed':s['physical_seed'],'regime':r,'repeat':q,'policy':p}
                     for s,r,q,_k,p in jobs],
        'reward':'negative measured 500-step run wall time including fix balance',
        'evaluation':'frozen, no exploration or updates, independent process/run',
        'script_sha256':digest(Path(__file__)),
    }
    if args.resume:
        if json.loads((output/'manifest.json').read_text()) != manifest:
            parser.error('resume code/config/model mismatch')
        results = (json.loads((output/'results.json').read_text())
                   if (output/'results.json').is_file() else [])
        ids = [f'seed-{s["physical_seed"]}-{r}-r{q}-{p}' for s,r,q,_k,p in jobs]
        if [row['id'] for row in results] != ids[:len(results)]:
            parser.error('result prefix mismatch')
    else:
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        results=[]
    for spec,regime,repeat,key,policy in jobs[len(results):]:
        attempt=output/f'seed-{spec["physical_seed"]}-{regime}-r{repeat}-{policy}'
        if attempt.exists():
            attempt.rename(attempt.with_name(attempt.name+f'-interrupted-{int(time.time())}'))
        result=run_episode(output,spec,policy,regime,repeat,
                           schedules[key]['active_by_decision'],models,
                           args.decisions,args.expected_atoms,cores)
        results.append(result)
        (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('COMPLETE',result['id'],result['total_seconds'],flush=True)
    print('ALL COMPLETE',len(results),flush=True)


if __name__ == '__main__':
    main()
