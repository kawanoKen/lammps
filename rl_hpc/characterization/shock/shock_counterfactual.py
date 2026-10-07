#!/usr/bin/env python3
"""Counterfactual neighbor-skin/load-balance study for shock/nemd.

The baseline creates restart checkpoints from one shock trajectory.  Every
measurement is a new MPI process reading exactly one checkpoint, performing a
fixed common 50-step warm-up (needed by ``balance ... weight time``), applying
one absolute skin/balance action, and timing the following 500 steps.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[3]
LMP = ROOT / "build_shock_char" / "lmp"
SOURCE = ROOT / "examples" / "PACKAGES" / "shock" / "nemd"
SKINS = (0.15, 0.25, 0.35, 0.5, 0.7, 1.0)
BALANCES = ("none", "atom", "neigh", "time")


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def run_command(command: list[str], cwd: Path, stdout: Path, stderr: Path, timeout: int = 900) -> tuple[int, float]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    start = perf_counter()
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out,
                                   stderr=err, text=True, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                for child in children:
                    try: child.terminate()
                    except psutil.NoSuchProcess: pass
                try: process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
                for child in children:
                    try:
                        if child.is_running(): child.kill()
                    except psutil.NoSuchProcess: pass
    return code, perf_counter() - start


def mpi_command(ranks: int, input_path: Path, log_path: Path) -> list[str]:
    return [
        "mpirun", "--bind-to", "core", "--map-by", "core", "-np", str(ranks),
        str(LMP), "-in", str(input_path), "-log", str(log_path), "-screen", "none",
    ]


def baseline_input(out: Path, checkpoint_steps: tuple[int, ...]) -> str:
    """Preserve the bundled in.nemd workload and segment only its final run."""
    source = (SOURCE / "in.nemd").read_text()
    lines: list[str] = []
    previous = 0
    for index, step in enumerate(checkpoint_steps, 1):
        lines += [
            f"run {step - previous}",
            f"write_restart {out / f'checkpoint-S{index}.restart'}",
            'print "SHOCK_CHECKPOINT S%d step=$(step) temp=$(temp) press=$(press) density=$(density) xshock=$(v_xshock)"' % index,
        ]
        previous = step
    marker = "run\t\t${nsteps}"
    if marker not in source:
        raise RuntimeError("unexpected in.nemd run marker")
    return source.replace(marker, "\n".join(lines), 1)


def balance_command(style: str) -> str | None:
    # 0.8 is the documented practical starting factor for neigh/time weights.
    if style == "none":
        return None
    suffix = {"atom": "", "neigh": " weight neigh 0.8", "time": " weight time 0.8"}[style]
    return "balance 1.0 shift x 10 1.0" + suffix


def fork_input(restart: Path, skin: float, balance: str, steps: int, warmup: int) -> str:
    command = balance_command(balance)
    lines = [
        # Input variables themselves are not restart-persistent; the shock
        # parameters are needed below only for the observable front position.
        f"include {restart.parent / 'shockparams.mod'}",
        f"read_restart {restart}",
        # fix nve and wall/reflect are not restart-persistent.  The original
        # NEMD trajectory used these exact fixes after the impact velocity.
        "fix mynve all nve",
        "fix xwalls all wall/reflect xlo EDGE xhi EDGE",
        "compute mymom all momentum",
        "variable xshock equal (1.0+c_mymom[1]/(atoms*${mass})/${up})*lx-${up}*time",
        "timer normal",
        # Identical preceding run for every action.  It establishes a current
        # neighbor list and supplies timer data to the time-weight action.
        "neighbor 0.5 bin",
        "neigh_modify every 1 delay 0 check yes",
        "thermo 100",
        "thermo_style custom step temp pe density etotal press v_xshock",
        f"run {warmup}",
        "variable action_start timer",
        f"neighbor {skin:.8g} bin",
        "neigh_modify every 1 delay 0 check yes",
        "run 0 post no",
    ]
    if command:
        lines.append(command)
    lines += [
        f"run {steps}",
        "variable action_stop timer",
        'print "ACTION_WALL_SECONDS $(v_action_stop-v_action_start:%.9f)"',
    ]
    return "\n".join(lines) + "\n"


def timing_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    lines = text.splitlines()
    row = re.compile(r"^\s*(Pair|Neigh|Comm|Modify|Other|Output|Kspace|KSpace)\s+\|\s+[^|]+\|\s+([0-9.eE+-]+)\s+\|")
    for i, line in enumerate(lines):
        match = re.match(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps", line)
        if not match:
            continue
        end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("Loop time of")), len(lines))
        block = "\n".join(lines[i:end])
        sections = {m.group(1).lower(): float(m.group(2)) for m in (row.match(x) for x in lines[i:end]) if m}
        def rank_stat(name: str) -> dict[str, float] | None:
            found = re.search(rf"{name}:\s+([0-9.eE+-]+) ave\s+([0-9.eE+-]+) max\s+([0-9.eE+-]+) min", block)
            return None if not found else {"mean": float(found.group(1)), "max": float(found.group(2)), "min": float(found.group(3))}
        blocks.append({
            "loop_seconds": float(match.group(1)), "mpi_ranks": int(match.group(2)), "steps": int(match.group(3)),
            "timing_avg_seconds": sections,
            "neighbor_builds": int((re.findall(r"Neighbor list builds =\s*(\d+)", block) or ["0"])[-1]),
            "dangerous_builds": int((re.findall(r"Dangerous builds =\s*(\d+)", block) or ["0"])[-1]),
            "nlocal": rank_stat("Nlocal"), "nghost": rank_stat("Nghost"), "neighs": rank_stat("Neighs"),
        })
    return blocks


def thermo(text: str) -> dict[str, float]:
    headers: list[str] = []
    latest: dict[str, float] = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "Step":
            headers = fields
        elif headers and len(fields) == len(headers):
            try:
                latest = {key: float(value) for key, value in zip(headers, fields)}
            except ValueError:
                pass
    return {"step": latest.get("Step", float("nan")), "temperature": latest.get("Temp", float("nan")),
            "pressure": latest.get("Press", float("nan")), "density": latest.get("Density", float("nan")),
            "xshock": latest.get("v_xshock", float("nan"))}


def parse_trial(log: Path) -> dict[str, Any]:
    text = log.read_text(errors="replace") if log.exists() else ""
    blocks = timing_blocks(text)
    action = re.findall(r"ACTION_WALL_SECONDS\s+([0-9.eE+-]+)", text)
    return {"measurement": blocks[-1] if blocks else {}, "warmup": blocks[-2] if len(blocks) > 1 else {},
            "thermo_after": thermo(text), "action_wall_seconds": float(action[-1]) if action else None,
            "errors": re.findall(r"^ERROR:.*$", text, re.MULTILINE), "warnings": re.findall(r"^WARNING:.*$", text, re.MULTILINE)}


def make_baseline(out: Path, ranks: int, checkpoint_steps: tuple[int, ...], nx: int = 240, ny: int = 12, nz: int = 12) -> list[dict[str, Any]]:
    params = (SOURCE / "shockparams.mod").read_text()
    for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
        params = re.sub(rf"(variable\s+{name}\s+index\s+)\d+", lambda match: match.group(1) + str(value), params)
    (out / "shockparams.mod").write_text(params)
    (out / "shocksetup.mod").write_text((SOURCE / "shocksetup.mod").read_text())
    script = out / "baseline.in"
    script.write_text(baseline_input(out, checkpoint_steps))
    log = out / "baseline.log"
    command = mpi_command(ranks, script, log)
    rc, wall = run_command(command, out, out / "baseline.stdout", out / "baseline.stderr", timeout=3600)
    if rc:
        raise RuntimeError(f"baseline failed with exit status {rc}; see {out / 'baseline.stderr'}")
    text = log.read_text(errors="replace")
    points = []
    pattern = r"^SHOCK_CHECKPOINT\s+(S\d+)\s+step=([^ ]+)\s+temp=([^ ]+)\s+press=([^ ]+)\s+density=([^ ]+)\s+xshock=([^\s]+)"
    for match in re.finditer(pattern, text, re.MULTILINE):
        label, step, temp, press, density, xshock = match.groups()
        points.append({"checkpoint_id": label, "step": float(step), "temperature": float(temp), "pressure": float(press),
                       "density": float(density), "xshock": float(xshock), "restart": str(out / f"checkpoint-{label}.restart")})
    if len(points) != len(checkpoint_steps):
        raise RuntimeError(f"expected {len(checkpoint_steps)} checkpoints, found {len(points)}")
    (out / "checkpoint_states.json").write_text(json.dumps(points, indent=2) + "\n")
    (out / "baseline_manifest.json").write_text(json.dumps({"ranks": ranks, "process_wall_seconds": wall, "command": shlex.join(command)}, indent=2) + "\n")
    return points


def trial(out: Path, checkpoint: dict[str, Any], ranks: int, skin: float, balance: str, repetition: int, steps: int, warmup: int) -> dict[str, Any]:
    ident = f"{checkpoint['checkpoint_id']}-np{ranks}-skin{skin:g}-{balance}-r{repetition}"
    directory = out / "runs" / ident
    directory.mkdir(parents=True, exist_ok=False)
    script = directory / "in.lammps"
    log = directory / "log.lammps"
    script.write_text(fork_input(Path(checkpoint["restart"]), skin, balance, steps, warmup))
    command = mpi_command(ranks, script, log)
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        rc, process_wall = run_command(command, directory, directory / "stdout.txt", directory / "stderr.txt")
    except subprocess.TimeoutExpired:
        rc, process_wall = -9, float("nan")
    parsed = parse_trial(log)
    metric = parsed["measurement"]
    numeric = [value for value in parsed["thermo_after"].values() if isinstance(value, float)]
    safe = rc == 0 and metric.get("dangerous_builds", 0) == 0 and not parsed["errors"] and all(value == value and abs(value) < 1.0e100 for value in numeric)
    return {"timestamp": timestamp, "git_commit": git_commit(), "workload": "shock/nemd", "checkpoint": checkpoint,
            "mpi_ranks": ranks, "omp_threads": 1, "skin": skin, "every": 1, "delay": 0, "check": "yes",
            "balance": balance, "weight_factor": None if balance in ("none", "atom") else 0.8,
            "repetition": repetition, "segment_steps": steps, "warmup_steps": warmup, "process_wall_seconds": process_wall,
            "lammps": parsed, "returncode": rc, "safe": safe, "command": shlex.join(command),
            "run_directory": str(directory.relative_to(ROOT))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ranks", type=int, default=32)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--checkpoints", type=int, default=6)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--preflight", action="store_true", help="S1 only; validates every action before main collection")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if not LMP.is_file():
        parser.error(f"missing required build: {LMP}")
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists() and any(out.iterdir()):
        parser.error(f"output must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # Shock moves in +x under this NEMD setup.  The 13,500-step endpoint is
    # deliberately well inside the 240-cell-long domain, avoiding reflection.
    checkpoint_steps = (1000, 3500, 6000, 8500, 11000, 13500)[:args.checkpoints]
    checkpoints = make_baseline(out, args.baseline_ranks, checkpoint_steps)
    selected = checkpoints[:1] if args.preflight else checkpoints
    jobs = [(cp, rank, skin, balance, rep) for cp in selected for rank in args.ranks for skin in SKINS for balance in BALANCES for rep in range(1, args.repetitions + 1)]
    random.Random(args.seed).shuffle(jobs)
    data = out / "measurements.jsonl"
    failures = 0
    with data.open("w") as stream:
        for index, (cp, rank, skin, balance, rep) in enumerate(jobs, 1):
            record = trial(out, cp, rank, skin, balance, rep, args.steps, args.warmup)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            failures += not record["safe"]
            loop = record["lammps"]["measurement"].get("loop_seconds")
            print(f"{index}/{len(jobs)} {cp['checkpoint_id']} np={rank} skin={skin:g} balance={balance} rep={rep} safe={record['safe']} loop={loop}", flush=True)
    manifest = {"git_commit": git_commit(), "baseline_ranks": args.baseline_ranks, "ranks": args.ranks, "checkpoint_steps": checkpoint_steps,
                "skins": SKINS, "balances": BALANCES, "repetitions": args.repetitions, "segment_steps": args.steps, "warmup_steps": args.warmup,
                "preflight": args.preflight, "jobs": len(jobs), "unsafe_or_failed": failures}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
