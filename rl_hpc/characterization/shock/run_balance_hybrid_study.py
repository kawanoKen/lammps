#!/usr/bin/env python3
"""Sequential resumable idle/CPU-jitter hybrid-balance training."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'rl_hpc/characterization/data'
SCRIPT = ROOT / 'rl_hpc/characterization/shock/online_balance_hybrid.py'
RESTARTS = DATA / 'shock_online_seed_restarts_v1/train_restart_list.json'


def main() -> None:
    for label, jitter in (('idle', False), ('cpu_jitter', True)):
        out = DATA / f'shock_balance_hybrid_{label}_v1'
        progress_path = out / 'progress.json'
        if progress_path.is_file() and json.loads(
                progress_path.read_text())['complete_episodes'] == 48:
            print('ALREADY COMPLETE', label, flush=True)
            continue
        command = [sys.executable, '-u', str(SCRIPT), '--output', str(out),
                   '--restart-list', str(RESTARTS), '--episodes', '48',
                   '--explore-episodes', '20', '--decisions', '120',
                   '--seed', '20261005']
        if jitter:
            command.extend(['--cpu-jitter', '--cpu-cores', '0-7'])
        if out.exists():
            command.append('--resume')
        print('START', label, 'resume=', out.exists(), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        print('COMPLETE', label, flush=True)


if __name__ == '__main__':
    main()
