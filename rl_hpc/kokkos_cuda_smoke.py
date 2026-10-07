#!/usr/bin/env python3
"""Run a small, auditable CUDA/KOKKOS smoke matrix.

The test deliberately keeps the CPU reference build and CUDA build separate.
The gpu/aware=on case is diagnostic: on a non-CUDA-aware MPI it is expected to
fail, and that failure is recorded rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def mpi_cuda_info() -> dict[str, object]:
    executable = shutil.which("ompi_info")
    if executable is None:
        return {"available": False, "reason": "ompi_info not found"}
    result = subprocess.run(
        [executable, "--all"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if "cuda" in line.lower() and any(
            key in line.lower()
            for key in ("built_with_cuda", "cuda_support", "mpi_cuda")
        )
    ]
    text = "\n".join(lines)
    def current_value(name: str) -> bool:
        match = re.search(
            rf"{re.escape(name)}.*?current value:\s*\"?(true|false)",
            text,
            flags=re.IGNORECASE,
        )
        return bool(match and match.group(1).lower() == "true")

    return {
        "available": result.returncode == 0,
        "cuda_support_lines": lines,
        "built_with_cuda_support": current_value("opal_built_with_cuda_support"),
        "cuda_support_enabled": current_value("opal_cuda_support"),
    }


def parse_thermo(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    rows: list[dict[str, float]] = []
    for index, line in enumerate(lines):
        header = line.strip().split()
        if not header or header[0] != "Step" or "TotEng" not in header:
            continue
        for data_line in lines[index + 1 :]:
            values = data_line.strip().split()
            if len(values) != len(header):
                if rows:
                    break
                continue
            try:
                converted = [float(value) for value in values]
            except ValueError:
                if rows:
                    break
                continue
            if not converted[0].is_integer():
                if rows:
                    break
                continue
            rows.append(dict(zip(header, converted)))
        if rows:
            break
    return rows


def parse_performance(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    result: dict[str, float] = {}
    loop = re.search(r"Loop time of\s+([0-9.eE+-]+)\s+on", text)
    performance = re.search(
        r"Performance:.*?([0-9.eE+-]+)\s+timesteps/s,\s+"
        r"([0-9.eE+-]+)\s+Matom-step/s",
        text,
    )
    if loop:
        result["lammps_loop_seconds"] = float(loop.group(1))
    if performance:
        result["timesteps_per_second"] = float(performance.group(1))
        result["matom_steps_per_second"] = float(performance.group(2))
    return result


def run_case(
    name: str,
    command: list[str],
    case_dir: Path,
    environment: dict[str, str],
    timeout_seconds: float,
) -> dict[str, object]:
    case_dir.mkdir(parents=True)
    (case_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    timed_out = False
    try:
        with (case_dir / "stdout.txt").open("w", encoding="utf-8") as stdout, (
            case_dir / "stderr.txt"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=case_dir,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_seconds,
                text=True,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        return_code = 124
    wall_seconds = time.perf_counter() - started
    log_path = case_dir / "log.lammps"
    return {
        "name": name,
        "command": command,
        "environment": {
            key: environment[key]
            for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")
            if key in environment
        },
        "return_code": return_code,
        "timed_out": timed_out,
        "success": return_code == 0 and not timed_out and log_path.exists(),
        "wall_seconds": wall_seconds,
        "lammps": parse_performance(log_path),
        "thermo_rows": len(parse_thermo(log_path)),
    }


def compare_thermo(cpu_log: Path, gpu_log: Path) -> dict[str, object]:
    cpu_rows = {int(row["Step"]): row for row in parse_thermo(cpu_log)}
    gpu_rows = {int(row["Step"]): row for row in parse_thermo(gpu_log)}
    common_steps = sorted(set(cpu_rows) & set(gpu_rows))
    fields = ["Temp", "PotEng", "KinEng", "TotEng", "Press"]
    max_abs = {field: 0.0 for field in fields}
    for step in common_steps:
        for field in fields:
            if field in cpu_rows[step] and field in gpu_rows[step]:
                difference = abs(cpu_rows[step][field] - gpu_rows[step][field])
                max_abs[field] = max(max_abs[field], difference)
    tolerance = 1.0e-8
    return {
        "cpu_steps": sorted(cpu_rows),
        "gpu1_steps": sorted(gpu_rows),
        "common_steps": common_steps,
        "max_absolute_difference": max_abs,
        "tolerance": tolerance,
        "consistent": bool(common_steps)
        and all(value <= tolerance for value in max_abs.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=87287)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl_hpc/runs/kokkos-cuda-smoke"),
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        parser.error(f"output already exists; choose a new directory: {output}")
    output.mkdir(parents=True)

    cpu_binary = root / "build_rl" / "lmp"
    gpu_binary = root / "build_kokkos_cuda" / "lmp"
    input_file = root / "rl_hpc" / "bench_lj.in"
    for required in (cpu_binary, gpu_binary, input_file):
        if not required.exists():
            parser.error(f"required file is missing: {required}")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        parser.error("mpirun was not found; the 4-GPU smoke test needs MPI")

    common_input = [
        "-var",
        "steps",
        str(args.steps),
        "-var",
        "seed",
        str(args.seed),
        "-in",
        str(input_file),
    ]
    base_environment = os.environ.copy()
    base_environment.setdefault("OMP_NUM_THREADS", "1")
    cases: list[dict[str, object]] = []

    cpu_command = [
        str(cpu_binary),
        "-log",
        "log.lammps",
        "-screen",
        "none",
        *common_input,
    ]
    cases.append(
        run_case(
            "cpu_reference",
            cpu_command,
            output / "cpu_reference",
            base_environment.copy(),
            args.timeout,
        )
    )

    for ranks, visible_devices in ((1, "0"), (4, "0,1,2,3")):
        for gpu_aware in ("off", "on"):
            name = f"gpu{ranks}_aware_{gpu_aware}"
            environment = base_environment.copy()
            environment["CUDA_VISIBLE_DEVICES"] = visible_devices
            command = [str(gpu_binary)]
            if ranks > 1:
                command = [mpirun, "-np", str(ranks), *command]
            command.extend(
                [
                    "-k",
                    "on",
                    "g",
                    "1",
                    "-sf",
                    "kk",
                    "-pk",
                    "kokkos",
                    "gpu/aware",
                    gpu_aware,
                    "-log",
                    "log.lammps",
                    "-screen",
                    "none",
                    *common_input,
                ]
            )
            cases.append(
                run_case(
                    name,
                    command,
                    output / name,
                    environment,
                    args.timeout,
                )
            )

    mpi_info = mpi_cuda_info()
    comparison = compare_thermo(
        output / "cpu_reference" / "log.lammps",
        output / "gpu1_aware_off" / "log.lammps",
    )
    for case in cases:
        case["accepted"] = bool(case["success"])
        if (
            case["name"] == "gpu4_aware_on"
            and not case["success"]
            and not mpi_info.get("cuda_support_enabled", False)
        ):
            case["accepted"] = True
            case["classification"] = "expected_failure_non_cuda_aware_mpi"

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "workload": "rl_hpc/bench_lj.in",
        "steps": args.steps,
        "seed": args.seed,
        "mpi_cuda_info": mpi_info,
        "numerical_comparison_cpu_vs_gpu1_off": comparison,
        "cases": cases,
        "success": bool(comparison["consistent"])
        and all(bool(case["accepted"]) for case in cases),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
