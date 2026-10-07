#!/usr/bin/env python3
"""Counterfactual balance-action forks from representative shock states."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

import collect_trajectories as collector
import shock_counterfactual as base


FACTORS = (None, 0.5, 0.75, 1.0, 1.25, 1.5)
CHILD: subprocess.Popen | None = None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_child() -> None:
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
            CHILD.wait(timeout=8)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(CHILD.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            CHILD.wait()
    CHILD = None


def input_text(record: dict, params: Path, factor: float | None) -> str:
    restart = Path(record["restart"])
    action = ("" if factor is None else
              f"balance 1.0 shift x 10 1.0 weight neigh {factor:g} out action.mesh")
    return f"""include {params}
read_restart {restart}
fix mynve all nve
fix xwalls all wall/reflect xlo EDGE xhi EDGE
compute mymom all momentum
variable xshock equal (1.0+c_mymom[1]/(atoms*${{mass}})/${{up}})*lx-${{up}}*time
timer normal
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
thermo 100
thermo_style custom step atoms temp pe density etotal press v_xshock
run 0 post no
# Reconstruct one common pre-action software state; excluded from reward.
balance 1.0 shift x 10 1.0 weight neigh 1.0 out reconstructed.mesh
run 0 post no
print "PRE_STATE $(step) $(atoms) $(temp) $(press) $(density) $(etotal) $(v_xshock)"
variable begin timer
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
run 0 post no
variable prepared timer
{action}
variable balanced timer
run 500
variable ended timer
print "TIMES $(v_ended-v_begin:%.9f) $(v_prepared-v_begin:%.9f) $(v_balanced-v_prepared:%.9f) $(v_ended-v_balanced:%.9f)"
{collector.OBS}
"""


def run_trial(output: Path, record: dict, params: Path,
              factor: float | None, repetition: int) -> dict:
    global CHILD
    label = "skip" if factor is None else f"factor-{factor:g}"
    key = f"phase-{record['phase']:02d}-{label}-r{repetition}"
    directory = output / "runs" / key
    directory.mkdir(parents=True, exist_ok=False)
    inp = directory / "in.lammps"
    inp.write_text(input_text(record, params, factor))
    log = directory / "log.lammps"
    command = base.mpi_command(32, inp, log)
    command.insert(1, "--nooversubscribe")
    started = time.time()
    rc = None
    try:
        with (directory / "stdout.txt").open("w") as stdout, \
                (directory / "stderr.txt").open("w") as stderr:
            CHILD = subprocess.Popen(
                command, cwd=directory, env={**os.environ, "OMP_NUM_THREADS": "1"},
                stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                rc = CHILD.wait(timeout=240)
            except subprocess.TimeoutExpired:
                rc = -9
    finally:
        clean_child()

    text = log.read_text(errors="replace") if log.exists() else ""
    matches = re.findall(r"^TIMES (.*)$", text, re.MULTILINE)
    times = [float(value) for value in matches[-1].split()] if matches else []
    pre = re.findall(r"^PRE_STATE (.*)$", text, re.MULTILINE)
    pre_values = [float(value) for value in pre[-1].split()] if pre else []
    blocks = [block for block in base.timing_blocks(text) if block["steps"] == 500]
    measurement = blocks[-1] if blocks else {}
    thermo_after = base.thermo(text)
    errors = re.findall(r"^ERROR:.*$", text, re.MULTILINE)
    safe = bool(
        rc == 0 and len(times) == 4 and len(pre_values) == 7
        and all(math.isfinite(value) and value >= 0 for value in times)
        and abs(times[0] - sum(times[1:])) <= 1e-6
        and measurement.get("steps") == 500
        and measurement.get("dangerous_builds") == 0
        and int(pre_values[0]) == int(record["step"])
        and int(thermo_after["step"]) == int(record["step"]) + 500
        and all(math.isfinite(value) for value in thermo_after.values())
        and not errors and "weight neigh skipped" not in text)
    row = {
        "key": key, "phase": record["phase"], "checkpoint_step": record["step"],
        "checkpoint_sha256": record["sha256"], "repetition": repetition,
        "action": {"balance": factor is not None, "factor": factor, "skin": 0.5},
        "pre_state_source": record["observed_state"],
        "pre_thermo_reconstructed": pre_values,
        "runtime_seconds": times[0] if times else None,
        "preparation_seconds": times[1] if times else None,
        "balance_seconds": times[2] if times else None,
        "run_seconds": times[3] if times else None,
        "reward": -times[0] if times else None,
        "next_state": {"thermo": thermo_after, "lammps": measurement},
        "safe": safe, "returncode": rc, "errors": errors,
        "timestamp": started, "process_wall_seconds": time.time() - started,
        "run_directory": str(directory),
    }
    (directory / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "main"), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    source_manifest = json.loads((args.checkpoints / "manifest.json").read_text())
    records = source_manifest["records"]
    params = Path(source_manifest["initial_restart"]["path"]).parent / "shockparams.mod"
    if not params.is_file():
        parser.error("shockparams.mod is missing")
    for record in records:
        restart = Path(record["restart"])
        if not restart.is_file() or digest(restart) != record["sha256"]:
            parser.error(f"missing or changed checkpoint: {restart}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {
        "version": "shock-representative-balance-forks-v1",
        "git_commit": base.git_commit(), "mpi_ranks": 32, "omp_threads": 1,
        "steps": 500, "skin": 0.5, "factors": list(FACTORS),
        "repetitions": args.repetitions, "random_seed": args.seed,
        "source_manifest": str((args.checkpoints / "manifest.json").resolve()),
        "source_manifest_sha256": digest(args.checkpoints / "manifest.json"),
        "pre_action_reconstruction": "balance weight neigh 1.0 outside timer",
        "reward": "negative action application plus 500-step wall time",
        "runner_sha256": digest(Path(__file__)),
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("manifest mismatch; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    data = output / "measurements.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines()] if data.exists() else []
    if any(not row["safe"] for row in rows):
        raise RuntimeError("unsafe existing trial requires audit")
    if args.phase == "smoke":
        jobs = [(records[index], factor, 0)
                for index in (0, len(records) - 1)
                for factor in (None, 0.5, 1.5)]
    else:
        jobs = [(record, factor, repetition)
                for repetition in range(args.repetitions)
                for record in records for factor in FACTORS]
        rng = random.Random(args.seed)
        block = len(records) * len(FACTORS)
        chunks = [jobs[index:index + block] for index in range(0, len(jobs), block)]
        for chunk in chunks:
            rng.shuffle(chunk)
        jobs = [job for chunk in chunks for job in chunk]
    done = {row["key"] for row in rows}
    with data.open("a") as stream:
        for record, factor, repetition in jobs:
            label = "skip" if factor is None else f"factor-{factor:g}"
            key = f"phase-{record['phase']:02d}-{label}-r{repetition}"
            if key in done:
                continue
            row = run_trial(output, record, params, factor, repetition)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(key, "safe", row["safe"], "seconds", row["runtime_seconds"], flush=True)
            if not row["safe"]:
                raise RuntimeError(f"unsafe or invalid trial: {key}")
            done.add(key)
    (output / f"{args.phase}_complete.json").write_text(json.dumps({
        "phase": args.phase, "trials": len(jobs), "timestamp": time.time()
    }, indent=2) + "\n")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        main()
    finally:
        clean_child()
