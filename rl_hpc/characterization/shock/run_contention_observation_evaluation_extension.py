#!/usr/bin/env python3
"""Add four held-out paired aware/blind conditions (n=4 to n=8)."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "rl_hpc/characterization/data"
SCRIPT = ROOT / "rl_hpc/characterization/shock/evaluate_contention_observation.py"
OUTPUT = DATA / "shock_contention_observation_eval_extension_v1"
RESTARTS = DATA / "shock_online_seed_restarts_v1/test_restart_list.json"


def main() -> None:
    command = [
        sys.executable, "-u", str(SCRIPT), "--output", str(OUTPUT),
        "--restart-list", str(RESTARTS), "--decisions", "120",
        "--repeat-start", "2", "--repeats", "2",
        "--policies", "aware", "blind",
    ]
    if OUTPUT.exists():
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
