#!/usr/bin/env python3
"""Create independent thermal-velocity shock starts without changing physics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import shock_counterfactual as base


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--original-restart', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int,
                   default=[17287, 27287, 37287, 47287, 57287])
    p.add_argument('--train-count', type=int, default=4,
                   help='first N restarts are for online training; remaining are held out')
    args = p.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or 87287 in args.seeds:
        p.error('seeds must be distinct and omit the original seed 87287')
    if not 1 <= args.train_count < len(args.seeds)+1:
        p.error('train-count must leave at least one held-out restart')
    original = args.original_restart.resolve()
    if not original.is_file():
        p.error('original restart not found')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    specs = [dict(path=str(original), physical_seed=87287,
                  sha256=digest(original))]
    script = Path(__file__).with_name('prepare_heavy.py')
    for seed in args.seeds:
        directory = out / f'seed-{seed}'
        subprocess.run([sys.executable, str(script), '--nx', '480',
                        '--ny', '16', '--nz', '16', '--velocity-seed', str(seed),
                        '--output', str(directory)], check=True)
        restart = directory / 'checkpoint-S1.restart'
        result = json.loads((directory / 'result.json').read_text())
        if result['return_code'] != 0 or result['atoms_expected'] != 491520:
            raise RuntimeError(f'invalid restart: {seed}')
        specs.append(dict(path=str(restart), physical_seed=seed,
                          sha256=digest(restart)))
        (out / 'restart_list.json').write_text(json.dumps(specs, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(
        dict(git_commit=base.git_commit(), workload='shock/nemd',
             nx=480, ny=16, nz=16, mpi_ranks=32,
             velocity_seeds=[entry['physical_seed'] for entry in specs],
             train_count=args.train_count,
             controlled_change='velocity all create temperature SEED only'),
        indent=2)+'\n')
    (out / 'train_restart_list.json').write_text(json.dumps(
        specs[:args.train_count], indent=2)+'\n')
    (out / 'test_restart_list.json').write_text(json.dumps(
        specs[args.train_count:], indent=2)+'\n')
    print('COMPLETE', len(specs), 'independent starts', flush=True)


if __name__ == '__main__':
    main()
