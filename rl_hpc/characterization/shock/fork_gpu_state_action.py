#!/usr/bin/env python3
"""GPU Shock checkpoint forks under idle, CPU, and direct GPU contention.

Each independent process reconstructs the same factor-1.0 partition, measures
a common 500-step pre-action probe under the selected resource regime, applies
one balance action, and measures the following 500 steps.  The probe supplies
an actually observed GPU performance state instead of borrowing CPU timings
from the trajectory that selected the checkpoint.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import threading
import time

import psutil

CHARACTERIZATION = Path(__file__).resolve().parents[1]
if str(CHARACTERIZATION) not in sys.path:
    sys.path.insert(0, str(CHARACTERIZATION))

import collect_trajectories as collector
import cpu_jitter
import fork_representative_balance_actions as forks
import gpu_factorial
import shock_counterfactual as base

ROOT = Path(__file__).resolve().parents[3]
GPU_LMP = ROOT / "build_shock_kokkos_cuda" / "lmp"
FACTORS = forks.FACTORS
REGIMES = ("idle", "cpu_contention", "gpu_contention")
GPU_IDS = tuple(x.strip() for x in os.environ.get(
    "LAMMPS_GPU_IDS", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
).split(",") if x.strip())
GPU_CONTENDER_CPUS = tuple(int(x.strip()) for x in os.environ.get(
    "LAMMPS_GPU_CONTENDER_CPUS", "40,41,42,43").split(",") if x.strip())
MPS_PIPE = f"/tmp/{os.environ.get('USER', 'user')}-lammps-mps-pipe"
MPS_LOG = f"/tmp/{os.environ.get('USER', 'user')}-lammps-mps-log"


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class GpuSimulation(collector.Simulation):
    """Interactive MPI32 process sharing four GPUs through CUDA MPS."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.command = [
            "mpirun", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
            "-np", "32", str(GPU_LMP), "-k", "on", "g", str(len(GPU_IDS)), "-sf", "kk",
            "-pk", "kokkos", "gpu/aware", "off", "-nonbuf", "-echo", "none",
            "-log", str(directory / "log.lammps"),
        ]
        self.stderr = (directory / "stderr.txt").open("w")
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(GPU_IDS),
            "CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE,
            "CUDA_MPS_LOG_DIRECTORY": MPS_LOG,
        }
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, text=True, bufsize=1, start_new_session=True,
            cwd=directory, env=env,
        )
        import queue
        self.queue = queue.Queue()
        self.index = 0

        def pump():
            for line in self.process.stdout:
                self.queue.put(line)
            self.queue.put(None)

        self.reader = threading.Thread(target=pump, daemon=True)
        self.reader.start()


class GpuContenders:
    """One audited CUDA workload on every GPU used by the target."""

    def __init__(self, binary: Path, intensity: dict[str, int]):
        self.binary = binary
        self.intensity = intensity
        self.processes: list[subprocess.Popen] = []

    def start(self):
        if len(GPU_CONTENDER_CPUS) != len(GPU_IDS):
            raise RuntimeError("LAMMPS_GPU_CONTENDER_CPUS must specify one CPU per visible GPU")
        for gpu, (gpu_id, cpu_id) in enumerate(zip(GPU_IDS, GPU_CONTENDER_CPUS)):
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE,
                "CUDA_MPS_LOG_DIRECTORY": MPS_LOG,
            }
            command = [
                "taskset", "-c", str(cpu_id), str(self.binary), "3600",
                str(self.intensity["buffer_mb"]), str(self.intensity["repeats"]),
                str(self.intensity["sleep_us"]),
            ]
            self.processes.append(subprocess.Popen(
                command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            ))
        time.sleep(0.5)
        if any(p.poll() is not None for p in self.processes):
            raise RuntimeError("GPU contender failed during startup")

    def close(self):
        outputs = []
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in self.processes:
            try:
                output, _ = process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                output, _ = process.communicate(timeout=5)
            outputs.append({"returncode": process.returncode, "output": output[-2000:]})
        return outputs


def action_label(factor):
    return "skip" if factor is None else f"factor-{factor:g}"


def key(record, factor, repetition, regime):
    return f"phase-{record['phase']:02d}-{action_label(factor)}-r{repetition}-{regime}"


def start_sampler(samples, stop):
    def sample():
        while not stop.is_set():
            samples.append({"timestamp": time.time(), "gpus": gpu_factorial.gpu_snapshot()})
            stop.wait(0.2)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return thread


def parse_times(text, label):
    matches = re.findall(rf"^{label} (.*)$", text, re.MULTILINE)
    if not matches:
        raise RuntimeError(f"missing {label}")
    values = [float(v) for v in matches[-1].split()]
    if not finite(values):
        raise RuntimeError(f"nonfinite {label}")
    return values


def run_trial(output, record, factor, repetition, regime, params, gpu_binary,
              intensity):
    identifier = key(record, factor, repetition, regime)
    directory = output / "runs" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    restart = Path(record["restart"])
    setup = f'''include {params}
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
balance 1.0 shift x 10 1.0 out reconstructed.mesh
run 0 post no
print "PRE_STATE $(step) $(atoms) $(temp) $(press) $(density) $(etotal) $(v_xshock)"
'''
    action = ("" if factor is None else
              f"balance 1.0 shift x 10 1.0 weight time {factor:g} out action.mesh")
    measured = f'''variable begin timer
{action}
variable balanced timer
run 500
variable ended timer
print "TIMES $(v_ended-v_begin:%.9f) 0.000000000 $(v_balanced-v_begin:%.9f) $(v_ended-v_balanced:%.9f)"
{collector.OBS}
'''
    (directory / "setup.lammps").write_text(setup)
    (directory / "measurement.lammps").write_text(measured)

    cpu = cpu_jitter.CpuJitter(range(8)) if regime == "cpu_contention" else None
    gpu = GpuContenders(gpu_binary, intensity) if regime == "gpu_contention" else None
    simulation = None
    samples = []
    sample_stop = threading.Event()
    sampler = start_sampler(samples, sample_stop)
    started = time.time()
    row = {
        "key": identifier, "phase": record["phase"], "checkpoint_step": record["step"],
        "checkpoint_sha256": record["sha256"], "repetition": repetition,
        "regime": regime, "backend": "kokkos_cuda", "mpi_ranks": 32,
        "omp_threads": 1, "gpu_count": len(GPU_IDS),
        "ranks_per_gpu": 32 // len(GPU_IDS),
        "action": {"balance": factor is not None, "weight": "time",
                   "factor": factor, "skin": 0.5},
        "safe": False, "timestamp": started, "errors": [],
    }
    try:
        if cpu:
            cpu.start()
        simulation = GpuSimulation(directory)
        setup_text = simulation.execute(setup)
        (directory / "setup.stdout.txt").write_text(setup_text)

        if cpu:
            cpu.set_active(True)
        if gpu:
            gpu.start()

        # The source state is measured under the current resource regime.
        probe = f'''variable probe_begin timer
run 500
variable probe_end timer
print "PROBE_SECONDS $(v_probe_end-v_probe_begin:%.9f)"
{collector.OBS}
'''
        probe_text = simulation.execute(probe)
        (directory / "probe.stdout.txt").write_text(probe_text)
        state = collector.observation(probe_text, 0.5, 1.0, expected_atoms=491520)
        probe_seconds = parse_times(probe_text, "PROBE_SECONDS")[0]

        worker_before = cpu.cpu_seconds() if cpu else 0.0
        action_text = simulation.execute(measured)
        worker_after = cpu.cpu_seconds() if cpu else 0.0
        (directory / "measurement.stdout.txt").write_text(action_text)
        next_state = collector.observation(action_text, 0.5, factor, expected_atoms=491520)
        times = parse_times(action_text, "TIMES")
        if len(times) != 4 or abs(times[0] - sum(times[1:])) > 1e-6:
            raise RuntimeError("invalid action timer identity")
        if next_state["step"] != state["step"] + 500:
            raise RuntimeError("non-500-step transition")

        row.update(
            source_state=state, source_probe_seconds=probe_seconds,
            reward=-times[0], runtime_seconds=times[0],
            preparation_seconds=times[1], balance_seconds=times[2], run_seconds=times[3],
            next_state=next_state, worker_cpu_seconds=worker_after-worker_before,
            gpu_telemetry=gpu_factorial.telemetry_summary(samples),
        )
        combined = setup_text + probe_text + action_text
        errors = re.findall(r"^ERROR.*$", combined, re.MULTILINE)
        if errors or not finite(row) or (cpu and row["worker_cpu_seconds"] <= 0):
            raise RuntimeError("unsafe/nonfinite trial")
        row["safe"] = True
    except BaseException as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        sample_stop.set()
        sampler.join(timeout=3)
        if cpu:
            cpu.close()
            row["cpu_workers_reaped"] = all(not p.is_alive() for p in cpu.children)
        if gpu:
            row["gpu_contenders"] = gpu.close()
            if any(item["returncode"] != -signal.SIGTERM
                   for item in row["gpu_contenders"]):
                row["safe"] = False
                row["errors"].append("GPU contender did not exit on controlled SIGTERM")
        if simulation:
            row["returncode"] = simulation.close(graceful=True)
        row["process_wall_seconds"] = time.time() - started
        (directory / "result.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "pilot", "main"), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--gpu-buffer-mb", type=int, default=128,
                        help="per-array CUDA contender allocation; tune using the ampb pilot")
    parser.add_argument("--gpu-repeats", type=int, default=32,
                        help="arithmetic iterations per element in the CUDA contender")
    parser.add_argument("--gpu-sleep-us", type=int, default=0,
                        help="idle microseconds after each contender kernel")
    args = parser.parse_args()
    if len(GPU_IDS) != 4 or len(GPU_CONTENDER_CPUS) != 4:
        parser.error("the current Shock GPU protocol requires four visible GPUs and four contender CPUs")
    if len(set(GPU_IDS)) != 4 or len(set(GPU_CONTENDER_CPUS)) != 4:
        parser.error("GPU IDs and contender CPUs must be unique")
    if args.gpu_buffer_mb < 8 or args.gpu_repeats < 1 or args.gpu_sleep_us < 0:
        parser.error("invalid GPU contender intensity")
    if not GPU_LMP.is_file():
        parser.error(f"missing GPU binary: {GPU_LMP}")
    source = args.checkpoints.resolve() / "manifest.json"
    manifest = json.loads(source.read_text())
    records = manifest["records"]
    if len(records) != 8:
        parser.error("expected eight representative checkpoints")
    for record in records:
        if forks.digest(Path(record["restart"])) != record["sha256"]:
            parser.error("checkpoint SHA256 mismatch")
    params = Path(manifest["initial_restart"]["path"]).parent / "shockparams.mod"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    lock = (output / "lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_binary = gpu_factorial.build_gpu_contender()
    intensity = {"buffer_mb": args.gpu_buffer_mb, "repeats": args.gpu_repeats,
                 "sleep_us": args.gpu_sleep_us}

    selected = [records[0], records[-1]] if args.phase in ("smoke", "pilot") else records
    factors = (None, 1.5) if args.phase in ("smoke", "pilot") else FACTORS
    repetitions = 1 if args.phase == "smoke" else args.repetitions
    jobs = [(r, f, rep, regime) for rep in range(repetitions)
            for r in selected for f in factors for regime in REGIMES]
    random.Random(args.seed).shuffle(jobs)
    protocol = {
        "version": "shock-gpu-state-action-v3", "git_commit": base.git_commit(),
        "binary_sha256": forks.digest(GPU_LMP), "source_manifest_sha256": forks.digest(source),
        "mpi_ranks": 32, "omp_threads": 1, "gpus": list(GPU_IDS),
        "ranks_per_gpu": 32 // len(GPU_IDS), "mps": True, "gpu_aware_mpi": False,
        "contender_cpus": list(GPU_CONTENDER_CPUS),
        "steps_probe": 500, "steps_action": 500, "skin": 0.5,
        "balance_weight": "time", "factors": FACTORS,
        "regimes": REGIMES, "repetitions": repetitions,
        "seed": args.seed, "schedule": [key(*job) for job in jobs],
        "state_groups": {
            "imbalance": ["nlocal", "nghost", "neighs", "atom_imbalance"],
            "performance": ["loop_seconds", "pair", "neigh", "comm", "modify", "other",
                            "neighbor_builds", "GPU utilization/power/clock/temperature"],
        },
        "contention": {
            "cpu": "8 compute workers on CPU logical IDs 0-7",
            "gpu": "one gpu_contender on each target GPU; direct MPS sharing",
            "gpu_contender_intensity": intensity,
            "cuda_arch": os.environ.get("GPU_CUDA_ARCH", "sm_86"),
        },
    }
    (output / "manifest.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with (output / "measurements.jsonl").open("a") as stream:
        for index, job in enumerate(jobs, 1):
            row = run_trial(output, *job, params=params, gpu_binary=gpu_binary,
                            intensity=intensity)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(index, len(jobs), row["key"], row["safe"], row.get("runtime_seconds"), flush=True)
            if not row["safe"]:
                raise RuntimeError(f"unsafe trial: {row['key']}: {row['errors']}")


if __name__ == "__main__":
    main()
