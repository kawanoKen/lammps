#!/usr/bin/env python3
"""Paired live evaluation of frozen continuous-skin policies, no updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

import cpu_jitter
import evaluate_online_cross as cross
import online_learn as old
import online_skin_continuous as skin
import shock_counterfactual as base

DATA = base.ROOT / 'rl_hpc/characterization/data'
POLICIES = ('skin_idle', 'skin_jitter', 'fixed_05_15', 'full_idle')
SOURCE = {'skin_idle': 'shock_skin_continuous_idle_v1',
          'skin_jitter': 'shock_skin_continuous_cpu_jitter_v1',
          'full_idle': 'shock_online_idle_main_v1'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models() -> tuple[dict, dict]:
    models, provenance = {}, {}
    for policy, dataset in SOURCE.items():
        folder = DATA / dataset
        manifest_path = folder / 'manifest.json'
        checkpoint = folder / 'latest_policy.npz'
        manifest = json.loads(manifest_path.read_text())
        progress = json.loads((folder / 'progress.json').read_text())
        if (progress['complete_episodes'] != 36 or
                progress['complete_transitions'] != 4320 or
                manifest['binary_sha256'] != digest(base.LMP) or
                manifest['decisions'] != 120):
            raise RuntimeError(f'incomplete or incompatible policy: {policy}')
        expected_feature = (old.FEATURE_VERSION if policy == 'full_idle'
                            else skin.FEATURE_VERSION)
        if manifest['feature_version'] != expected_feature:
            raise RuntimeError(f'feature mismatch: {policy}')
        with np.load(checkpoint, allow_pickle=False) as data:
            if str(data['feature_version']) != expected_feature:
                raise RuntimeError(f'checkpoint feature mismatch: {policy}')
            coef = np.array(data['coef'], copy=True)
        if not np.isfinite(coef).all():
            raise RuntimeError(f'nonfinite checkpoint: {policy}')
        models[policy] = coef
        provenance[policy] = {
            'dataset': dataset, 'feature_version': expected_feature,
            'checkpoint_sha256': digest(checkpoint),
            'manifest_sha256': digest(manifest_path),
            'training_physical_seeds': [r['physical_seed'] for r in manifest['restarts']]}
    return models, provenance


def choose(policy: str, state: dict, decision: int, models: dict) -> tuple[float, float]:
    if policy == 'fixed_05_15':
        return (.5, 1.5)
    if policy == 'full_idle':
        return cross.select('idle_trained', state, decision,
                            {'idle_trained': models['full_idle']})
    learner = skin.SkinFQI(120)
    learner.coef = models[policy]
    return (learner.choose(state), skin.FACTOR)


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
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.decisions <= 120 or args.repeats < 1:
        parser.error('decisions must be 1..120; repeats >= 1')
    if len(set(args.regimes)) != len(args.regimes) or len(set(args.policies)) != len(args.policies):
        parser.error('duplicate regime or policy')
    models, provenance = load_models()
    specs = json.loads(args.restart_list.read_text())
    if not specs:
        parser.error('empty restart list')
    training_seeds = {seed for item in provenance.values()
                      for seed in item['training_physical_seeds']}
    for spec in specs:
        path = Path(spec['path']).resolve()
        if spec['physical_seed'] in training_seeds:
            parser.error('evaluation restart was used in training')
        if not path.is_file() or digest(path) != spec['sha256']:
            parser.error(f'missing or changed restart: {path}')
    cores = cpu_jitter.parse_cores(args.cpu_cores) if 'jitter' in args.regimes else ()
    out = args.output.resolve()
    if args.resume:
        if not (out / 'manifest.json').is_file():
            parser.error('resume requires an existing manifest')
    else:
        out.mkdir(parents=True, exist_ok=False)
    schedules, jobs = {}, []
    for repeat in range(args.repeats):
        for spec in specs:
            for regime in args.regimes:
                key = f'{spec["physical_seed"]}-{regime}-r{repeat}'
                schedule_seed = args.seed + 1009 * repeat + spec['physical_seed']
                schedule = (cpu_jitter.schedule(args.decisions, schedule_seed, 2, 5)
                            if regime == 'jitter' else [False] * args.decisions)
                schedules[key] = {'seed': schedule_seed,
                                  'active_by_decision': schedule}
                policies = list(args.policies)
                random.Random(schedule_seed ^ 0xA11CE).shuffle(policies)
                jobs.extend((spec, regime, repeat, key, policy) for policy in policies)
    manifest = {
        'version': 'shock-continuous-skin-heldout-v1',
        'git_commit': base.git_commit(), 'lammps_binary': str(base.LMP),
        'lammps_sha256': digest(base.LMP),
        'restart_list': str(args.restart_list.resolve()), 'heldout_restarts': specs,
        'frozen_policies': provenance, 'policies': args.policies,
        'regimes': args.regimes, 'decisions': args.decisions,
        'steps_per_decision': 500, 'repeats': args.repeats,
        'expected_atoms': args.expected_atoms, 'mpi_ranks': 32,
        'omp_threads': 1, 'cpu_cores': list(cores), 'schedule': schedules,
        'run_order': [{'seed': spec['physical_seed'], 'regime': regime,
                       'repeat': repeat, 'policy': policy}
                      for spec, regime, repeat, _, policy in jobs],
        'reward': 'negative action+run wall seconds; warmup/startup excluded',
        'evaluation': 'frozen; no online update or exploration; one process per run',
        'fixed_baseline': 'skin 0.5, neighbor factor 1.5, balance every interval',
        'caveat': 'held-out seeds were used in earlier full-action evaluation, '
                  'but not in any policy training',
        'script_sha256': digest(Path(__file__)),
        'shared_runner_sha256': digest(Path(cross.__file__)),
        'skin_policy_code_sha256': digest(Path(skin.__file__)),
    }
    if args.resume:
        previous = json.loads((out / 'manifest.json').read_text())
        if previous != manifest:
            parser.error('resume configuration, models or code differ from manifest')
        results = (json.loads((out / 'results.json').read_text())
                   if (out / 'results.json').is_file() else [])
        expected_ids = [f'seed-{spec["physical_seed"]}-{regime}-r{repeat}-{policy}'
                        for spec, regime, repeat, _, policy in jobs]
        if [item['id'] for item in results] != expected_ids[:len(results)]:
            parser.error('completed results do not match run order')
    else:
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        results = []
    for spec, regime, repeat, key, policy in jobs[len(results):]:
        attempt = out / f'seed-{spec["physical_seed"]}-{regime}-r{repeat}-{policy}'
        if attempt.exists():
            if not args.resume:
                raise RuntimeError(f'unexpected output {attempt}')
            preserved = attempt.with_name(attempt.name + f'-interrupted-{int(time.time())}')
            attempt.rename(preserved)
            print('PRESERVED', preserved, flush=True)
        result = cross.episode(out, spec, policy, regime, repeat,
                               schedules[key]['active_by_decision'], models,
                               args.decisions, args.expected_atoms, cores,
                               selector=choose)
        results.append(result)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('COMPLETE', result['id'], result['total_seconds'], flush=True)
    print('ALL COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    main()
