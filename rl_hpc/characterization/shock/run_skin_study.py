#!/usr/bin/env python3
"""Sequential, resumable idle/CPU-jitter training with matched experience."""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'rl_hpc/characterization/data'
SCRIPT = ROOT / 'rl_hpc/characterization/shock/online_skin_continuous.py'
RESTARTS = DATA / 'shock_online_seed_restarts_v1/train_restart_list.json'


def main() -> None:
    for label, jitter in (('idle', False), ('cpu_jitter', True)):
        out = DATA / f'shock_skin_continuous_{label}_v1'
        progress_path = out / 'progress.json'
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            if progress['complete_episodes'] == 36:
                print('ALREADY COMPLETE', label, flush=True)
                continue
        command = [sys.executable, '-u', str(SCRIPT),
                   '--output', str(out), '--restart-list', str(RESTARTS),
                   '--episodes', '36', '--explore-episodes', '12',
                   '--decisions', '120', '--seed', '20261004']
        if jitter:
            command.extend(['--cpu-jitter', '--cpu-cores', '0-7'])
        if out.exists():
            command.append('--resume')
        print('START', label, 'resume=', out.exists(), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        print('COMPLETE', label, flush=True)


if __name__ == '__main__':
    main()
