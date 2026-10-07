#!/usr/bin/env python3
"""Collect a compact workload x state x action characterization matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CHAR_DIR = ROOT / "rl_hpc" / "characterization"
WORKLOAD_DIR = CHAR_DIR / "workloads"
RAW_DEFAULT = CHAR_DIR / "raw"
CO_RUNNER = CHAR_DIR / "co_runner.py"
GPU_SOURCE = CHAR_DIR / "gpu_contender.cu"
GPU_BINARY = CHAR_DIR / "build" / "gpu_contender"


@dataclass(frozen=True)
class Action:
    name: str
    skin: str
    every: int
    delay: int = 0
    check: str = "yes"

    def variables(self) -> dict[str, str]:
        return {
            "skin": self.skin,
            "neigh_every": str(self.every),
            "neigh_delay": str(self.delay),
            "neigh_check": self.check,
        }


WORKLOADS: dict[str, dict[str, Any]] = {
    "lj": {
        "input": WORKLOAD_DIR / "lj_scaled.in",
        "binary": ROOT / "build_rl_char" / "lmp",
        "mpi": 4,
        "omp": 1,
        "steps": 100,
        "seed": 87287,
        "actions": (
            Action("lj_skin03_every1", "0.3", 1),
            Action("lj_skin06_every5", "0.6", 5),
            Action("lj_skin10_every20", "1.0", 20),
        ),
    },
    "spce": {
        "input": WORKLOAD_DIR / "spce.in",
        "binary": ROOT / "build_rl_char" / "lmp",
        "mpi": 4,
        "omp": 1,
        "steps": 100,
        "seed": 432567,
        "extra_variables": {"datafile": str(ROOT / "bench/POTENTIALS/data.spce")},
        "actions": (
            Action("spce_skin20_every1", "2.0", 1),
            Action("spce_skin30_every5", "3.0", 5),
            Action("spce_skin40_every10", "4.0", 10),
        ),
    },
}


def command_output(command: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def git_commit() -> str:
    return command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])


def system_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        snapshot["loadavg_1_5_15"] = list(os.getloadavg())
    except OSError:
        snapshot["loadavg_1_5_15"] = []
    try:
        meminfo: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw_value = line.split(":", 1)
            meminfo[key] = int(raw_value.strip().split()[0])
        snapshot["mem_available_kb"] = meminfo.get("MemAvailable")
        snapshot["mem_free_kb"] = meminfo.get("MemFree")
    except (OSError, ValueError):
        pass
    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=3,
    )
    snapshot["gpu"] = gpu
    return snapshot


def parse_log(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {}
    text = log_path.read_text(errors="replace")
    result: dict[str, Any] = {}
    loop_matches = re.findall(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+([0-9]+)", text)
    if loop_matches:
        result["loop_seconds"] = float(loop_matches[-1][0])
        result["loop_mpi_ranks"] = int(loop_matches[-1][1])
    performance = re.findall(r"^Performance:\s*(.+)$", text, re.MULTILINE)
    if performance:
        line = performance[-1]
        result["performance_line"] = line
        for key, pattern in {
            "timesteps_per_second": r"([0-9.eE+-]+)\s+timesteps/s",
            "matom_steps_per_second": r"([0-9.eE+-]+)\s+Matom-step/s",
            "katom_steps_per_second": r"([0-9.eE+-]+)\s+katom-step/s",
            "tau_per_day": r"([0-9.eE+-]+)\s+tau/day",
            "ns_per_day": r"([0-9.eE+-]+)\s+ns/day",
        }.items():
            match = re.search(pattern, line)
            if match:
                result[key] = float(match.group(1))
    timing: dict[str, float] = {}
    row_pattern = re.compile(
        r"^\s*(Pair|Bond|Neigh|Kspace|KSpace|Comm|Output|Modify|Other)\s+\|\s+"
        r"([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s+\|"
    )
    for line in text.splitlines():
        match = row_pattern.match(line)
        if match:
            timing[match.group(1).lower()] = float(match.group(3))
    if timing:
        result["timing_avg_seconds"] = timing
    builds = re.findall(r"Neighbor list builds =\s*([0-9]+)", text)
    dangerous = re.findall(r"Dangerous builds =\s*([0-9]+)", text)
    if builds:
        result["neighbor_builds"] = int(builds[-1])
    if dangerous:
        result["dangerous_builds"] = int(dangerous[-1])
    atoms = re.findall(r"Loop time of.*?for\s+([0-9]+)\s+atoms", text)
    if atoms:
        result["atoms"] = int(atoms[-1])
    return result


def build_gpu_contender() -> Path:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc is required for the GPU contention probe")
    GPU_BINARY.parent.mkdir(parents=True, exist_ok=True)
    command = [nvcc, "-O3", "-arch=sm_86", str(GPU_SOURCE), "-o", str(GPU_BINARY)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GPU contender build failed:\n{result.stderr[-4000:]}")
    return GPU_BINARY


def start_contender(state: str, duration: float, gpu_binary: Path | None) -> subprocess.Popen[str] | None:
    if state == "idle":
        return None
    if state == "cpu_contention":
        command = [
            "taskset", "-c", "0-3", sys.executable, str(CO_RUNNER), "cpu",
            "--duration", str(duration), "--workers", "4",
        ]
    elif state == "memory_contention":
        command = [
            "taskset", "-c", "8-15", sys.executable, str(CO_RUNNER), "memory",
            "--duration", str(duration), "--size-mb", "512",
        ]
    elif state == "gpu_contention":
        if gpu_binary is None:
            raise RuntimeError("GPU contender was not built")
        command = [str(gpu_binary), str(duration)]
    else:
        raise ValueError(f"unknown contention state: {state}")
    return subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})


def stop_contender(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def run_trial(
    raw_root: Path,
    workload_name: str,
    workload: dict[str, Any],
    action: Action,
    state: str,
    repetition: int,
    purpose: str,
    backend: str = "cpu",
    steps_override: int | None = None,
    gpu_binary: Path | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    run_id = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}-{workload_name}-{backend}-{state}-{action.name}-r{repetition}"
    run_dir = raw_root / run_id
    run_dir.mkdir(parents=True)
    log_path = run_dir / "log.lammps"
    binary = workload["binary"] if backend == "cpu" else ROOT / "build_kokkos_cuda" / "lmp"
    steps = steps_override or workload["steps"]
    variables = {
        "steps": str(steps),
        "seed": str(workload["seed"]),
        **action.variables(),
        **workload.get("extra_variables", {}),
    }
    command: list[str]
    if backend == "cpu":
        command = [
            "mpirun", "--bind-to", "core", "--map-by", "core", "-np",
            str(workload["mpi"]), str(binary),
        ]
    else:
        command = [
            str(binary), "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos",
            "gpu/aware", "off",
        ]
    command += ["-log", str(log_path), "-screen", "none"]
    for name, value in variables.items():
        command += ["-var", name, value]
    command += ["-in", str(workload["input"])]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(workload["omp"])
    if backend == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    before = system_snapshot()
    contender = None
    process_start = time.perf_counter()
    try:
        contender = start_contender(state, 45.0, gpu_binary)
        if contender is not None:
            time.sleep(0.4)
        with (run_dir / "stdout.txt").open("w") as stdout, (run_dir / "stderr.txt").open("w") as stderr:
            completed = subprocess.run(
                command,
                cwd=run_dir,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
                text=True,
                timeout=40,
            )
        process_wall = time.perf_counter() - process_start
    except subprocess.TimeoutExpired:
        completed = None
        process_wall = time.perf_counter() - process_start
    finally:
        stop_contender(contender)
    after = system_snapshot()
    telemetry = parse_log(log_path)
    success = completed is not None and completed.returncode == 0 and "loop_seconds" in telemetry
    safe = success and telemetry.get("dangerous_builds", 0) == 0
    record: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "git_commit": git_commit(),
        "run_id": run_id,
        "purpose": purpose,
        "workload": workload_name,
        "backend": backend,
        "contention_state": state,
        "repetition": repetition,
        "mpi_ranks": workload["mpi"] if backend == "cpu" else 1,
        "omp_threads": workload["omp"],
        "gpu_count": 1 if backend == "gpu" else 0,
        "action": {"name": action.name, **action.variables()},
        "segment_steps": steps,
        "process_wall_seconds": process_wall,
        "lammps": telemetry,
        "system_before": before,
        "system_after": after,
        "exit_status": None if completed is None else completed.returncode,
        "success": success,
        "scientifically_safe": safe,
        "raw_directory": str(run_dir.relative_to(ROOT)),
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CHAR_DIR / "data")
    parser.add_argument("--matrix-reps", type=int, default=3)
    parser.add_argument("--stability-reps", type=int, default=5)
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    if args.matrix_reps < 1 or args.stability_reps < 1:
        parser.error("repetition counts must be positive")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        parser.error(f"output is non-empty; choose a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_root = (output.parent / "raw")
    raw_root.mkdir(parents=True, exist_ok=True)
    data_path = output / "measurements.jsonl"
    gpu_binary = None
    if not args.skip_gpu:
        gpu_binary = build_gpu_contender()
    records: list[dict[str, Any]] = []

    def collect(record: dict[str, Any]) -> None:
        records.append(record)
        with data_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        status = "ok" if record["scientifically_safe"] else "FAIL"
        loop = record["lammps"].get("loop_seconds", "-")
        print(f"{status:4s} {record['workload']:5s} {record['backend']:4s} {record['contention_state']:18s} {record['action']['name']:22s} r{record['repetition']} loop={loop}", flush=True)

    for workload_name, workload in WORKLOADS.items():
        baseline = workload["actions"][0]
        for repetition in range(1, args.stability_reps + 1):
            collect(run_trial(raw_root, workload_name, workload, baseline, "idle", repetition, "stability", gpu_binary=gpu_binary))
        states = ["idle", "cpu_contention", "memory_contention"]
        for state in states:
            for action in workload["actions"]:
                for repetition in range(1, args.matrix_reps + 1):
                    collect(run_trial(raw_root, workload_name, workload, action, state, repetition, "cpu_state_action_matrix", gpu_binary=gpu_binary))

    if not args.skip_gpu:
        gpu_workload = WORKLOADS["lj"]
        for state in ("idle", "gpu_contention"):
            for action in gpu_workload["actions"]:
                for repetition in range(1, args.matrix_reps + 1):
                    collect(run_trial(raw_root, "lj", gpu_workload, action, state, repetition, "gpu_state_action_matrix", backend="gpu", steps_override=500, gpu_binary=gpu_binary))

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "workloads": {name: {key: str(value) for key, value in info.items() if key in ("input", "binary", "mpi", "omp", "steps", "seed")} for name, info in WORKLOADS.items()},
        "matrix_reps": args.matrix_reps,
        "stability_reps": args.stability_reps,
        "record_count": len(records),
        "gpu_contender": None if gpu_binary is None else str(gpu_binary),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if all(record["success"] and record["scientifically_safe"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
