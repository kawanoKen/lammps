#!/usr/bin/env python3
"""Run one auditable LAMMPS experiment and write a JSON summary.

This is deliberately a small process launcher.  It is not an RL environment
and does not tune parameters automatically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = SCRIPT_DIR / "bench_lj.in"
DEFAULT_LAMMPS = REPO_ROOT / "build_rl" / "lmp"


def command_output(argv: list[str], cwd: Path | None = None) -> str:
    """Return short command output, or an empty string if unavailable."""
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def first_matching_line(path: Path, prefix: str) -> str:
    try:
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(prefix):
                return line.strip()
    except OSError:
        pass
    return ""


def machine_metadata(lammps_binary: Path) -> dict[str, Any]:
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            meminfo[key] = value.strip()
    except (OSError, ValueError):
        pass

    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader",
        ]
    )
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_model": first_matching_line(Path("/proc/cpuinfo"), "model name"),
        "mem_total": meminfo.get("MemTotal", ""),
        "git_commit": command_output(["git", "rev-parse", "HEAD"], REPO_ROOT),
        "git_describe": command_output(
            ["git", "describe", "--tags", "--always", "--dirty"], REPO_ROOT
        ),
        "gpu_query": gpu,
        "lammps_binary": str(lammps_binary),
    }


def parse_lammps_log(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {}
    text = log_path.read_text(errors="replace")
    loop_matches = re.findall(r"Loop time of\s+([0-9.eE+-]+)", text)
    performance_lines = re.findall(r"^Performance:\s*(.+)$", text, re.MULTILINE)
    parsed: dict[str, Any] = {}
    if loop_matches:
        parsed["loop_time_seconds"] = float(loop_matches[-1])
    if performance_lines:
        line = performance_lines[-1]
        parsed["performance_line"] = line
        patterns = {
            "timesteps_per_second": r"([0-9.eE+-]+)\s+timesteps/s",
            "matom_step_per_second": r"([0-9.eE+-]+)\s+Matom-step/s",
            "tau_per_day": r"([0-9.eE+-]+)\s+tau/day",
            "ns_per_day": r"([0-9.eE+-]+)\s+ns/day",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                parsed[key] = float(match.group(1))
    return parsed


def parse_lammps_var(raw: str) -> tuple[str, str]:
    name, separator, value = raw.partition("=")
    if not separator or not name or not value or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise argparse.ArgumentTypeError(
            f"invalid --lammps-var {raw!r}; expected NAME=VALUE"
        )
    return name, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpi", type=int, default=1, help="number of MPI processes")
    parser.add_argument("--steps", type=int, default=1000, help="LAMMPS timesteps")
    parser.add_argument("--seed", type=int, default=87287, help="velocity random seed")
    parser.add_argument(
        "--omp-threads", type=int, default=1, help="OMP_NUM_THREADS per MPI rank"
    )
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="LAMMPS input")
    parser.add_argument("--lammps", type=Path, default=DEFAULT_LAMMPS, help="LAMMPS executable")
    parser.add_argument(
        "--lammps-var",
        action="append",
        default=[],
        type=parse_lammps_var,
        metavar="NAME=VALUE",
        help="additional index-style LAMMPS variable; repeatable",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mpi < 1 or args.steps < 1 or args.omp_threads < 1:
        raise SystemExit("--mpi, --steps, and --omp-threads must be positive")

    executable = args.lammps.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SystemExit(f"LAMMPS executable is not runnable: {executable}")
    if not input_path.is_file():
        raise SystemExit(f"LAMMPS input does not exist: {input_path}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)

    variables: list[tuple[str, str]] = [("steps", str(args.steps)), ("seed", str(args.seed))]
    supplied_names = {name for name, _ in args.lammps_var}
    if supplied_names.intersection({"steps", "seed"}):
        raise SystemExit("do not override steps or seed with --lammps-var; use their options")
    variables.extend(args.lammps_var)

    command: list[str] = []
    if args.mpi == 1:
        command.append(str(executable))
    else:
        launcher = os.environ.get("MPI_LAUNCHER") or shutil.which("mpirun")
        if not launcher:
            raise SystemExit("--mpi > 1 requested, but no mpirun was found")
        command.extend([launcher, "-np", str(args.mpi), str(executable)])
    command.extend(["-log", "log.lammps", "-screen", "screen.out"])
    for name, value in variables:
        command.extend(["-var", name, value])
    command.extend(["-in", str(input_path)])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    started = dt.datetime.now(dt.timezone.utc)
    start_monotonic = time.perf_counter()
    with (output / "stdout.txt").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
        completed = subprocess.run(
            command,
            cwd=output,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
    wall_seconds = time.perf_counter() - start_monotonic

    result: dict[str, Any] = {
        "timestamp": started.isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"], REPO_ROOT),
        "workload": "bundled_lj_fixed_32000",
        "input": str(input_path.relative_to(REPO_ROOT))
        if input_path.is_relative_to(REPO_ROOT)
        else str(input_path),
        "mpi_processes": args.mpi,
        "omp_threads_per_mpi_rank": args.omp_threads,
        "lammps_parameters": {name: value for name, value in variables},
        "command": command,
        "wall_clock_seconds": wall_seconds,
        "exit_status": completed.returncode,
        "success": completed.returncode == 0,
        "lammps_performance": parse_lammps_log(output / "log.lammps"),
        "machine": machine_metadata(executable),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if completed.returncode != 0:
        print(f"LAMMPS failed with exit status {completed.returncode}; see {output}", file=sys.stderr)
        return completed.returncode if completed.returncode > 0 else 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("interrupted")
