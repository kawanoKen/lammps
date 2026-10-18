#!/usr/bin/env python3
"""Resumable matched blind/aware CPU-contention training study."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "rl_hpc/characterization/data"
SCRIPT = ROOT / "rl_hpc/characterization/shock/online_balance_contention.py"
RESTARTS = DATA / "shock_online_seed_restarts_v1/train_restart_list.json"


def main() -> None:
    # Their only intended difference is whether the three CPU-utilization
    # observations are exposed to the learner.
    for label, aware in (("aware", True), ("blind", False)):
        output = DATA / f"shock_balance_contention_{label}_v1"
        progress = output / "progress.json"
        if progress.is_file() and json.loads(
                progress.read_text())["complete_episodes"] == 48:
            print("ALREADY COMPLETE", label, flush=True)
            continue
        command = [
            sys.executable, "-u", str(SCRIPT), "--output", str(output),
            "--restart-list", str(RESTARTS), "--episodes", "48",
            "--explore-episodes", "20", "--decisions", "120",
            "--seed", "20261014", "--cpu-cores", "0-7",
            "--block-min-seconds", "45", "--block-max-seconds", "90",
        ]
        if aware:
            command.append("--contention-observation")
        if output.exists():
            command.append("--resume")
        print("START", label, "resume=", output.exists(), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        print("COMPLETE", label, flush=True)


if __name__ == "__main__":
    main()
