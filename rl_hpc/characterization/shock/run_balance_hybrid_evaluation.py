#!/usr/bin/env python3
"""Run/resume the full hybrid-balance held-out evaluation and analyze it."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SHOCK = ROOT / 'rl_hpc/characterization/shock'
DATA = ROOT / 'rl_hpc/characterization/data'
OUTPUT = DATA / 'shock_balance_hybrid_eval_main_v1'
RESTARTS = DATA / 'shock_online_seed_restarts_v1/test_restart_list.json'


def main() -> None:
    evaluator = SHOCK / 'evaluate_balance_hybrid.py'
    command = [sys.executable, '-u', str(evaluator), '--output', str(OUTPUT),
               '--restart-list', str(RESTARTS), '--decisions', '120',
               '--repeats', '2', '--regimes', 'idle', 'jitter',
               '--policies', 'idle_trained', 'jitter_trained', 'always_skip',
               'fixed_15', 'threshold_15']
    if OUTPUT.exists():
        command.append('--resume')
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable,
                    str(SHOCK / 'analyze_balance_hybrid_evaluation.py'),
                    str(OUTPUT)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
