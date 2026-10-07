#!/usr/bin/env python3
"""Measure the GPU skin x every factorial under controlled contention.

This is deliberately separate from the original three-action characterization
runner.  It keeps the target on GPU 0, randomizes action order within every
state/repetition, and records both LAMMPS timing and sampled nvidia-smi data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from characterize import ROOT, WORKLOAD_DIR, parse_log, git_commit


CHAR_DIR = ROOT / "rl_hpc" / "characterization"
CO_RUNNER = CHAR_DIR / "co_runner.py"
GPU_SOURCE = CHAR_DIR / "gpu_contender.cu"
GPU_BINARY = CHAR_DIR / "build" / "gpu_contender"


@dataclass(frozen=True)
class Action:
    name: str
    skin: str
    every: int

    def variables(self) -> dict[str, str]:
        return {
            "skin": self.skin,
            "neigh_every": str(self.every),
            "neigh_delay": "0",
            "neigh_check": "yes",
        }


LJ_ACTIONS = tuple(
    Action(f"lj_skin{skin.replace('.', '')}_every{every}", skin, every)
    for skin in ("0.3", "0.6", "1.0")
    for every in (1, 5, 20)
)
SPCE_ACTIONS = tuple(
    Action(f"spce_skin{skin.replace('.', '')}_every{every}", skin, every)
    for skin in ("2.0", "3.0", "4.0")
    for every in (1, 5, 10)
)


INTENSITIES: dict[str, dict[str, int]] = {
    # buffer_mb is per array; the contender allocates two arrays.
    "light": {"buffer_mb": 64, "repeats": 8, "sleep_us": 1000},
    "medium": {"buffer_mb": 128, "repeats": 32, "sleep_us": 0},
    "heavy": {"buffer_mb": 256, "repeats": 64, "sleep_us": 0},
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


def gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,clocks.mem,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    text = command_output(command, timeout=3.0)
    fields = (
        "index", "utilization_gpu_percent", "utilization_memory_percent",
        "memory_used_mb", "memory_total_mb", "clock_sm_mhz", "clock_mem_mhz",
        "power_w", "temperature_c",
    )
    result: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(fields):
            continue
        row: dict[str, Any] = {}
        for field, value in zip(fields, parts):
            try:
                row[field] = int(float(value)) if field == "index" else float(value)
            except ValueError:
                row[field] = value
        result.append(row)
    return result


def telemetry_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        for row in sample.get("gpus", []):
            if isinstance(row.get("index"), int):
                by_gpu[row["index"]].append(row)
    summary: dict[str, Any] = {}
    numeric = (
        "utilization_gpu_percent", "utilization_memory_percent", "memory_used_mb",
        "clock_sm_mhz", "clock_mem_mhz", "power_w", "temperature_c",
    )
    for gpu, rows in sorted(by_gpu.items()):
        values: dict[str, Any] = {"samples": len(rows)}
        for field in numeric:
            numbers = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
            if numbers:
                values[field] = {
                    "mean": sum(numbers) / len(numbers),
                    "max": max(numbers),
                    "min": min(numbers),
                }
        summary[str(gpu)] = values
    return summary


def sample_telemetry(stop: threading.Event, samples: list[dict[str, Any]], period: float) -> None:
    while not stop.is_set():
        samples.append({"timestamp": datetime.now(timezone.utc).isoformat(), "gpus": gpu_snapshot()})
        stop.wait(period)


def build_gpu_contender() -> Path:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc is required for GPU contention")
    GPU_BINARY.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [nvcc, "-O3", "-arch=sm_86", str(GPU_SOURCE), "-o", str(GPU_BINARY)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GPU contender build failed:\n{result.stderr[-4000:]}")
    return GPU_BINARY


def contention_spec(state: str) -> tuple[str, str | None, dict[str, int] | None]:
    if state == "idle":
        return "idle", None, None
    if state.startswith("same_gpu_"):
        intensity = state.removeprefix("same_gpu_")
        return "same_gpu", "0", INTENSITIES[intensity]
    if state == "different_gpu_medium":
        return "different_gpu", "1", INTENSITIES["medium"]
    if state == "cpu_only":
        return "cpu", None, None
    raise ValueError(f"unknown contention state: {state}")


def start_contender(state: str, duration: float, binary: Path | None) -> subprocess.Popen[str] | None:
    kind, visible_gpu, intensity = contention_spec(state)
    if kind == "idle":
        return None
    if kind == "cpu":
        command = [
            "taskset", "-c", "0-3", sys.executable, str(CO_RUNNER), "cpu",
            "--duration", str(duration), "--workers", "4",
        ]
        env = os.environ.copy()
    else:
        if binary is None or intensity is None or visible_gpu is None:
            raise RuntimeError("GPU contender configuration is incomplete")
        # Keep the CUDA launch thread away from the target's CPUs (0--3).
        # GPUs 0 and 1 are local to NUMA node 0, so CPUs 8--11 preserve the
        # same NUMA locality without introducing direct CPU-core contention.
        command = [
            "taskset", "-c", "8-11", str(binary), str(duration), str(intensity["buffer_mb"]),
            str(intensity["repeats"]), str(intensity["sleep_us"]),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible_gpu}
    return subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_contender(process: subprocess.Popen[str] | None) -> tuple[int | None, str]:
    if process is None:
        return None, ""
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        output, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate(timeout=5)
    return process.returncode, output or ""


def run_trial(
    output: Path,
    workload: str,
    action: Action,
    state: str,
    repetition: int,
    steps: int,
    contender_binary: Path,
    gpu_build: Path,
    sample_period: float,
    order_index: int,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    run_id = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}-{workload}-gpu-{state}-{action.name}-r{repetition}"
    run_dir = output / run_id
    run_dir.mkdir(parents=True)
    log_path = run_dir / "log.lammps"
    input_path = WORKLOAD_DIR / ("lj_scaled.in" if workload == "lj" else "spce.in")
    action_vars = action.variables()
    variables = {
        "steps": str(steps),
        "seed": "87287" if workload == "lj" else "432567",
        **action_vars,
    }
    if workload == "spce":
        variables["datafile"] = str(ROOT / "bench/POTENTIALS/data.spce")
    command = [
        "taskset", "-c", "0-3", str(gpu_build),
        "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "gpu/aware", "off",
        "-log", str(log_path), "-screen", "none",
    ]
    for name, value in variables.items():
        command.extend(["-var", name, value])
    command.extend(["-in", str(input_path)])
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1"}
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    before = gpu_snapshot()
    contender = None
    samples: list[dict[str, Any]] = []
    sampler_stop = threading.Event()
    sampler = threading.Thread(target=sample_telemetry, args=(sampler_stop, samples, sample_period), daemon=True)
    process_start = time.perf_counter()
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        contender = start_contender(state, 90.0, contender_binary)
        if contender is not None:
            time.sleep(0.5)
        sampler.start()
        with (run_dir / "stdout.txt").open("w") as stdout, (run_dir / "stderr.txt").open("w") as stderr:
            completed = subprocess.run(
                command,
                cwd=run_dir,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=90,
                text=True,
            )
    except subprocess.TimeoutExpired:
        completed = None
    finally:
        sampler_stop.set()
        sampler.join(timeout=5)
        contender_status, contender_output = stop_contender(contender)
    process_wall = time.perf_counter() - process_start
    after = gpu_snapshot()
    (run_dir / "contender.stdout").write_text(contender_output, encoding="utf-8")
    (run_dir / "gpu_telemetry.jsonl").write_text(
        "\n".join(json.dumps(sample, sort_keys=True) for sample in samples) + ("\n" if samples else ""),
        encoding="utf-8",
    )
    lammps = parse_log(log_path)
    success = completed is not None and completed.returncode == 0 and "loop_seconds" in lammps
    safe = success and lammps.get("dangerous_builds", 0) == 0
    kind, contender_gpu, intensity = contention_spec(state)
    record = {
        "timestamp": timestamp.isoformat(),
        "git_commit": git_commit(),
        "run_id": run_id,
        "workload": workload,
        "backend": "gpu",
        "contention_state": state,
        "contention_type": kind,
        "contention_intensity": state.removeprefix("same_gpu_") if state.startswith("same_gpu_") else ("medium" if state == "different_gpu_medium" else "none"),
        "gpu_mapping": {"target_physical_gpu": 0, "contender_visible_gpu": contender_gpu},
        "contender_config": intensity,
        "contender_exit_status": contender_status,
        "repetition": repetition,
        "randomized_order_index": order_index,
        "mpi_ranks": 1,
        "omp_threads": 1,
        "gpu_count": 1,
        "action": {"name": action.name, **action_vars},
        "segment_steps": steps,
        "process_wall_seconds": process_wall,
        "lammps": lammps,
        "gpu_telemetry_before": before,
        "gpu_telemetry_after": after,
        "gpu_telemetry_samples": samples,
        "gpu_telemetry_summary": telemetry_summary(samples),
        "exit_status": None if completed is None else completed.returncode,
        "success": success,
        "scientifically_safe": safe,
        "raw_directory": str(run_dir.relative_to(ROOT)),
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", choices=("lj", "spce"), default="lj")
    parser.add_argument("--gpu-build", type=Path, default=ROOT / "build_kokkos_cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--sample-period", type=float, default=0.25)
    parser.add_argument("--states", nargs="+", default=("idle", "same_gpu_light", "same_gpu_medium", "same_gpu_heavy", "different_gpu_medium", "cpu_only"))
    parser.add_argument("--actions", nargs="*", help="explicit action names; default is the complete 3x3 grid")
    args = parser.parse_args()
    if args.steps < 1 or args.reps < 1 or args.sample_period <= 0:
        parser.error("steps/reps must be positive and sample-period must be positive")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        parser.error(f"output is non-empty: {output}")
    gpu_build = args.gpu_build if args.gpu_build.is_absolute() else ROOT / args.gpu_build
    # The documented interface accepts either the build directory or the
    # executable itself.  Resolve the former once, rather than letting
    # taskset attempt to execute a directory during every trial.
    if gpu_build.is_dir():
        gpu_build = gpu_build / "lmp"
    if not gpu_build.is_file() or not os.access(gpu_build, os.X_OK):
        parser.error(f"GPU executable not found or not executable: {gpu_build}")
    if not gpu_snapshot():
        parser.error("nvidia-smi returned no usable GPU telemetry; refusing to start a factorial with an unavailable driver")
    output.mkdir(parents=True, exist_ok=True)
    contender_binary = build_gpu_contender()
    actions = LJ_ACTIONS if args.workload == "lj" else SPCE_ACTIONS
    by_name = {action.name: action for action in actions}
    if args.actions:
        unknown = [name for name in args.actions if name not in by_name]
        if unknown:
            parser.error(f"unknown actions: {', '.join(unknown)}")
        actions = tuple(by_name[name] for name in args.actions)
    all_states = set(args.states)
    valid_states = {"idle", "same_gpu_light", "same_gpu_medium", "same_gpu_heavy", "different_gpu_medium", "cpu_only"}
    if not all_states <= valid_states:
        parser.error(f"unknown state(s): {', '.join(sorted(all_states - valid_states))}")
    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    data_path = output / "measurements.jsonl"
    for repetition in range(1, args.reps + 1):
        for state in args.states:
            order = list(actions)
            rng.shuffle(order)
            for order_index, action in enumerate(order):
                record = run_trial(
                    output, args.workload, action, state, repetition, args.steps,
                    contender_binary, gpu_build, args.sample_period, order_index,
                )
                records.append(record)
                with data_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                status = "ok" if record["scientifically_safe"] else "FAIL"
                loop = record["lammps"].get("loop_seconds", "-")
                print(f"{status:4s} {args.workload:5s} {state:20s} {action.name:24s} r{repetition} loop={loop}", flush=True)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "workload": args.workload,
        "gpu_build": str(gpu_build),
        "steps": args.steps,
        "repetitions": args.reps,
        "states": list(args.states),
        "actions": [action.__dict__ for action in actions],
        "random_seed": args.seed,
        "record_count": len(records),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if all(record["success"] and record["scientifically_safe"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
