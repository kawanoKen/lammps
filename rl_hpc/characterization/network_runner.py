#!/usr/bin/env python3
"""Launch an allocated multi-node network co-runner and LAMMPS target.

This launcher refuses to infer hosts.  The caller must pass nodes belonging to
the current allocation, for example ``--hosts nodeA,nodeB``.  It is prepared
for a real allocation and is not used with localhost as a network proxy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import shlex
import signal
import subprocess
import time
from typing import Any

from characterize import ROOT, parse_log, git_commit
from gpu_factorial import SPCE_ACTIONS, Action


CHAR_DIR = ROOT / "rl_hpc" / "characterization"
CORUNNER = CHAR_DIR / "network_corunner" / "mpi_stream"


NETWORK_STATES: dict[str, dict[str, int]] = {
    "N0_idle": {"message_mb": 0, "streams": 0, "sleep_us": 0},
    "N1_light": {"message_mb": 4, "streams": 1, "sleep_us": 2000},
    "N2_medium": {"message_mb": 16, "streams": 1, "sleep_us": 500},
    "N3_heavy": {"message_mb": 32, "streams": 2, "sleep_us": 0},
}


def mpirun_command(hosts: str, ranks: int, executable: str, args: list[str]) -> list[str]:
    return ["mpirun", "--host", hosts, "-np", str(ranks), executable, *args]


def run_case(
    output: Path,
    hosts: str,
    target_ranks: int,
    corunner_ranks: int,
    binary: Path,
    input_file: Path,
    action: Action,
    state: str,
    steps: int,
    repetition: int,
    order_index: int,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    run_id = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}-spce-{state}-{action.name}-r{repetition}"
    run_dir = output / run_id
    run_dir.mkdir(parents=True)
    log_path = run_dir / "log.lammps"
    variables = {
        "steps": str(steps),
        "seed": "432567",
        "datafile": str(ROOT / "bench/POTENTIALS/data.spce"),
        **action.variables(),
    }
    target_args = [
        "-log", str(log_path), "-screen", "none",
    ]
    for name, value in variables.items():
        target_args.extend(["-var", name, value])
    target_args.extend(["-in", str(input_file)])
    target = mpirun_command(hosts, target_ranks, str(binary), target_args)
    spec = NETWORK_STATES[state]
    traffic = None
    if spec["message_mb"] > 0:
        traffic = mpirun_command(
            hosts,
            corunner_ranks,
            str(CORUNNER),
            [
                "--seconds", "90",
                "--message-mb", str(spec["message_mb"]),
                "--streams", str(spec["streams"]),
                "--sleep-us", str(spec["sleep_us"]),
            ],
        )
    (run_dir / "target_command.txt").write_text(shlex.join(target) + "\n", encoding="utf-8")
    if traffic:
        (run_dir / "corunner_command.txt").write_text(shlex.join(traffic) + "\n", encoding="utf-8")
    env = os.environ.copy()
    contender = None
    contender_out = ""
    started = time.perf_counter()
    try:
        if traffic:
            contender = subprocess.Popen(traffic, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            time.sleep(0.5)
        with (run_dir / "stdout.txt").open("w") as stdout, (run_dir / "stderr.txt").open("w") as stderr:
            completed = subprocess.run(target, cwd=run_dir, stdout=stdout, stderr=stderr, check=False, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        completed = None
    finally:
        if contender is not None:
            if contender.poll() is None:
                contender.send_signal(signal.SIGTERM)
            try:
                contender_out, _ = contender.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                contender.kill()
                contender_out, _ = contender.communicate(timeout=10)
    process_wall = time.perf_counter() - started
    (run_dir / "corunner_stdout.txt").write_text(contender_out or "", encoding="utf-8")
    throughput = None
    match = re.search(r"aggregate_gib_per_s=([0-9.eE+-]+)", contender_out or "")
    if match:
        throughput = float(match.group(1))
    lammps = parse_log(log_path)
    success = completed is not None and completed.returncode == 0 and "loop_seconds" in lammps
    return {
        "timestamp": timestamp.isoformat(),
        "git_commit": git_commit(),
        "run_id": run_id,
        "workload": "spce",
        "backend": "cpu_mpi",
        "network_state": state,
        "network_config": spec,
        "hosts": hosts.split(","),
        "target_mpi_ranks": target_ranks,
        "corunner_mpi_ranks": corunner_ranks if traffic else 0,
        "action": {"name": action.name, **action.variables()},
        "segment_steps": steps,
        "randomized_order_index": order_index,
        "process_wall_seconds": process_wall,
        "lammps": lammps,
        "corunner_throughput_gib_per_s": throughput,
        "target_exit_status": None if completed is None else completed.returncode,
        "corunner_exit_status": None if contender is None else contender.returncode,
        "success": success,
        "raw_directory": str(run_dir.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", required=True, help="comma-separated nodes in the current allocation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build_rl_char" / "lmp")
    parser.add_argument("--input", type=Path, default=CHAR_DIR / "workloads" / "spce.in")
    parser.add_argument("--target-ranks", type=int, default=8)
    parser.add_argument("--corunner-ranks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--states", nargs="+", choices=tuple(NETWORK_STATES), default=tuple(NETWORK_STATES))
    parser.add_argument("--actions", nargs="*", help="explicit SPC/E action names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_ranks < 1 or args.corunner_ranks < 1 or args.steps < 1 or args.reps < 1:
        parser.error("ranks, steps, and reps must be positive")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    binary = args.binary if args.binary.is_absolute() else ROOT / args.binary
    input_file = args.input if args.input.is_absolute() else ROOT / args.input
    if not CORUNNER.exists() and not args.dry_run:
        parser.error(f"network co-runner is not built: {CORUNNER}; run mpicc -O3 ... first")
    actions_by_name = {action.name: action for action in SPCE_ACTIONS}
    actions = SPCE_ACTIONS
    if args.actions:
        unknown = [name for name in args.actions if name not in actions_by_name]
        if unknown:
            parser.error(f"unknown actions: {', '.join(unknown)}")
        actions = tuple(actions_by_name[name] for name in args.actions)
    preview = mpirun_command(args.hosts, args.corunner_ranks, str(CORUNNER), ["--seconds", "90", "--message-mb", "16", "--streams", "1", "--sleep-us", "500"])
    if args.dry_run:
        print(json.dumps({"hosts": args.hosts.split(","), "target_example": mpirun_command(args.hosts, args.target_ranks, str(binary), ["-in", str(input_file)]), "corunner_example": preview, "states": args.states, "actions": [action.name for action in actions]}, indent=2))
        return 0
    if output.exists() and any(output.iterdir()):
        parser.error(f"output is non-empty: {output}")
    output.mkdir(parents=True)
    rng = random.Random(args.seed)
    data_path = output / "measurements.jsonl"
    records: list[dict[str, Any]] = []
    for repetition in range(1, args.reps + 1):
        for state in args.states:
            order = list(actions)
            rng.shuffle(order)
            for order_index, action in enumerate(order):
                record = run_case(output, args.hosts, args.target_ranks, args.corunner_ranks, binary, input_file, action, state, args.steps, repetition, order_index)
                records.append(record)
                with data_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(f"{'ok' if record['success'] else 'FAIL':4s} {state} {action.name} r{repetition} loop={record['lammps'].get('loop_seconds', '-')}", flush=True)
    manifest = {"timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "hosts": args.hosts.split(","), "record_count": len(records), "states": list(args.states), "actions": [action.name for action in actions], "steps": args.steps, "repetitions": args.reps}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if all(record["success"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
