#!/usr/bin/env python3
"""Run or resume the complete held-out continuous-skin policy evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'rl_hpc/characterization/data'
OUTPUT = DATA / 'shock_skin_continuous_eval_main_v1'
SCRIPT = ROOT / 'rl_hpc/characterization/shock/evaluate_skin_continuous.py'
RESTARTS = DATA / 'shock_online_seed_restarts_v1/test_restart_list.json'


def main() -> None:
    results = OUTPUT / 'results.json'
    if results.is_file() and len(json.loads(results.read_text())) == 32:
        print('ALREADY COMPLETE 32 runs', flush=True)
        return
    command = [sys.executable, '-u', str(SCRIPT),
               '--output', str(OUTPUT), '--restart-list', str(RESTARTS),
               '--decisions', '120', '--repeats', '2',
               '--regimes', 'idle', 'jitter',
               '--policies', 'skin_idle', 'skin_jitter',
               'fixed_05_15', 'full_idle']
    if OUTPUT.exists():
        command.append('--resume')
    print('START evaluation resume=', OUTPUT.exists(), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
