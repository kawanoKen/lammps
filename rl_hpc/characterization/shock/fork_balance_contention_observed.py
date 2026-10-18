#!/usr/bin/env python3
"""Counterfactual factor sweep with pre-action CPU/application observations.

Every branch reconstructs the same partition, runs a common 500-step probe
under a fixed contention regime, then applies one balance action and measures
the following 500 steps.  Probe telemetry is available before the action;
the regime label and co-runner counters are analysis-only ground truth.
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
import statistics
import time

import psutil

import collect_trajectories as collector
import cpu_host_monitor
import cpu_jitter
import cpu_resource_observer
import fork_representative_balance_actions as forks
import shock_counterfactual as base


FACTORS = (None, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
PROBE_STEPS = 500
ACTION_STEPS = 500


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def label(factor: float | None) -> str:
    return "skip" if factor is None else f"factor-{factor:g}"


def key(record: dict, factor: float | None, repetition: int, regime: str) -> str:
    return f"phase-{record['phase']:02d}-{label(factor)}-r{repetition}-{regime}"


def process_snapshot(processes: list[psutil.Process]) -> dict[int, dict]:
    result = {}
    for process in processes:
        try:
            cpu = process.cpu_times()
            switches = process.num_ctx_switches()
            result[process.pid] = {
                "cpu_seconds": cpu.user + cpu.system,
                "voluntary_switches": switches.voluntary,
                "involuntary_switches": switches.involuntary,
                "affinity": process.cpu_affinity(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def physical_siblings(cores: tuple[int, ...]) -> tuple[int, ...]:
    wanted = set()
    identities = set()
    for core in cores:
        root = Path(f"/sys/devices/system/cpu/cpu{core}/topology")
        identities.add((int((root / "physical_package_id").read_text()),
                        int((root / "core_id").read_text())))
    for core in range(psutil.cpu_count(logical=True) or 0):
        root = Path(f"/sys/devices/system/cpu/cpu{core}/topology")
        identity = (int((root / "physical_package_id").read_text()),
                    int((root / "core_id").read_text()))
        if identity in identities:
            wanted.add(core)
    return tuple(sorted(wanted))


def rank_summary(before: dict[int, dict], after: dict[int, dict], wall: float,
                 target_physical_cpus: tuple[int, ...], total_busy_seconds: float) -> dict:
    efficiencies = []
    voluntary = []
    involuntary = []
    target_rank_cpu = 0.0
    target_set = set(target_physical_cpus)
    for pid in sorted(set(before) & set(after)):
        cpu_delta = max(0.0, after[pid]["cpu_seconds"] - before[pid]["cpu_seconds"])
        efficiencies.append(cpu_delta / wall)
        voluntary.append(max(0, after[pid]["voluntary_switches"]
                             - before[pid]["voluntary_switches"]) / wall)
        involuntary.append(max(0, after[pid]["involuntary_switches"]
                               - before[pid]["involuntary_switches"]) / wall)
        if target_set.intersection(after[pid]["affinity"]):
            target_rank_cpu += cpu_delta
    if len(efficiencies) != 32:
        raise RuntimeError(f"expected 32 persistent LAMMPS ranks, got {len(efficiencies)}")

    def describe(values: list[float]) -> dict:
        return {
            "mean": statistics.fmean(values), "minimum": min(values),
            "maximum": max(values), "std": statistics.pstdev(values),
        }

    return {
        "rank_cpu_efficiency": describe(efficiencies),
        "rank_voluntary_context_switches_per_second": describe(voluntary),
        "rank_involuntary_context_switches_per_second": describe(involuntary),
        "target_physical_busy_core_seconds": total_busy_seconds,
        "target_lammps_rank_cpu_seconds": target_rank_cpu,
        "external_busy_core_seconds_estimate": max(0.0, total_busy_seconds - target_rank_cpu),
    }


def busy_seconds(before: cpu_resource_observer.Snapshot,
                 after: cpu_resource_observer.Snapshot,
                 cores: tuple[int, ...]) -> float:
    result = 0.0
    for core in cores:
        first, last = before.cpu_times[core], after.cpu_times[core]
        total = sum(getattr(last, field) - getattr(first, field)
                    for field in first._fields)
        idle = last.idle - first.idle
        if "iowait" in first._fields:
            idle += last.iowait - first.iowait
        result += max(0.0, total - idle)
    return result


def setup_commands(record: dict, params: Path) -> str:
    return f"""include {params}
read_restart {Path(record['restart'])}
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
balance 1.0 shift x 10 1.0 weight neigh 1.0 out reconstructed.mesh
run 0 post no
print "PRE_STATE $(step) $(atoms) $(temp) $(press) $(density) $(etotal) $(v_xshock)"
"""


def probe_commands() -> str:
    return f"""neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
run {PROBE_STEPS}
{collector.OBS}
"""


def action_commands(factor: float | None) -> str:
    action = ("" if factor is None else
              f"balance 1.0 shift x 10 1.0 weight neigh {factor:g} out action.mesh")
    return f"""variable begin timer
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
run 0 post no
variable prepared timer
{action}
variable balanced timer
run {ACTION_STEPS}
variable ended timer
print "TIMES $(v_ended-v_begin:%.9f) $(v_prepared-v_begin:%.9f) $(v_balanced-v_prepared:%.9f) $(v_ended-v_balanced:%.9f)"
{collector.OBS}
"""


def lammps_ranks(simulation: collector.Simulation) -> list[psutil.Process]:
    ranks = [process for process in psutil.Process(simulation.process.pid).children(recursive=True)
             if process.name() == "lmp"]
    if len(ranks) != 32:
        raise RuntimeError(f"expected 32 lmp ranks, found {len(ranks)}")
    return ranks


def run_trial(output: Path, record: dict, params: Path, factor: float | None,
              repetition: int, regime: str, contention_cores: tuple[int, ...],
              target_physical_cpus: tuple[int, ...]) -> dict:
    identifier = key(record, factor, repetition, regime)
    directory = output / "runs" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "setup.lammps").write_text(setup_commands(record, params))
    (directory / "probe.lammps").write_text(probe_commands())
    (directory / "action.lammps").write_text(action_commands(factor))
    contender = cpu_jitter.CpuJitter(contention_cores) if regime == "contention" else None
    simulation = None
    row = {
        "key": identifier, "phase": record["phase"],
        "checkpoint_step": record["step"], "checkpoint_sha256": record["sha256"],
        "repetition": repetition, "regime": regime,
        "action": {"balance": factor is not None, "factor": factor, "skin": 0.5},
        "safe": False, "errors": [], "timestamp": time.time(),
    }
    started = time.monotonic()
    try:
        if contender:
            contender.start()
        simulation = collector.Simulation(directory)
        setup_text = simulation.execute(setup_commands(record, params))
        (directory / "setup.stdout.txt").write_text(setup_text)
        pre_values = list(map(float, re.findall(
            r"^PRE_STATE (.*)$", setup_text, re.M)[-1].split()))
        ranks = lammps_ranks(simulation)
        bindings = [{"pid": process.pid, "cpus": process.cpu_affinity()}
                    for process in ranks]
        occupied = set().union(*(set(item["cpus"]) for item in bindings))
        if not set(contention_cores).issubset(occupied):
            raise RuntimeError("contention cores do not overlap the MPI placement")

        if contender:
            contender.set_active(True)
            time.sleep(0.15)
        observer = cpu_resource_observer.CpuResourceObserver(
            target_physical_cpus, mpi_ranks=32)
        resource_before = observer.snapshot(simulation.process.pid)
        rank_before = process_snapshot(ranks)
        worker_before = contender.cpu_seconds() if contender else 0.0
        probe_text = simulation.execute(probe_commands())
        worker_after_probe = contender.cpu_seconds() if contender else 0.0
        rank_after = process_snapshot(ranks)
        resource_after = observer.snapshot(simulation.process.pid)
        (directory / "probe.stdout.txt").write_text(probe_text)
        probe_state = collector.observation(
            probe_text, 0.5, 1.0, expected_atoms=491520)
        probe_resource = observer.summarize(resource_before, resource_after)
        probe_resource.update(rank_summary(
            rank_before, rank_after,
            resource_after.monotonic - resource_before.monotonic,
            target_physical_cpus,
            busy_seconds(resource_before, resource_after, target_physical_cpus)))

        action_resource_before = observer.snapshot(simulation.process.pid)
        action_rank_before = process_snapshot(ranks)
        action_text = simulation.execute(action_commands(factor))
        action_rank_after = process_snapshot(ranks)
        action_resource_after = observer.snapshot(simulation.process.pid)
        worker_after_action = contender.cpu_seconds() if contender else 0.0
        (directory / "action.stdout.txt").write_text(action_text)
        next_factor = factor if factor is not None else 1.0
        next_state = collector.observation(
            action_text, 0.5, next_factor, expected_atoms=491520)
        action_resource = observer.summarize(action_resource_before,
                                             action_resource_after)
        action_resource.update(rank_summary(
            action_rank_before, action_rank_after,
            action_resource_after.monotonic - action_resource_before.monotonic,
            target_physical_cpus,
            busy_seconds(action_resource_before, action_resource_after,
                         target_physical_cpus)))
        times = list(map(float, re.findall(r"^TIMES (.*)$", action_text, re.M)[-1].split()))
        if contender:
            contender.set_active(False)

        errors = re.findall(r"^ERROR.*$", setup_text + probe_text + action_text, re.M)
        row.update({
            "pre_thermo_reconstructed": pre_values,
            "reconstructed_mesh_sha256": forks.digest(directory / "reconstructed.mesh"),
            "mpi_bindings": bindings,
            "state_before_action": probe_state,
            "pre_action_resource_state": probe_resource,
            "analysis_ground_truth": {
                "contention_label": regime,
                "probe_corunner_cpu_seconds": worker_after_probe - worker_before,
                "action_corunner_cpu_seconds": worker_after_action - worker_after_probe,
            },
            "runtime_seconds": times[0], "reward": -times[0],
            "preparation_seconds": times[1], "balance_seconds": times[2],
            "run_seconds": times[3], "next_state": next_state,
            "post_action_resource_state": action_resource,
            "errors": errors,
        })
        if (len(pre_values) != 7 or int(pre_values[0]) != int(record["step"])
                or probe_state["step"] != record["step"] + PROBE_STEPS
                or next_state["step"] != record["step"] + PROBE_STEPS + ACTION_STEPS
                or len(times) != 4 or min(times) < 0
                or abs(times[0] - sum(times[1:])) > 1e-6
                or probe_state["previous_segment"]["dangerous_builds"] != 0
                or next_state["previous_segment"]["dangerous_builds"] != 0
                or errors or not finite(row)
                or (contender and row["analysis_ground_truth"]["probe_corunner_cpu_seconds"] <= 0)):
            raise RuntimeError("unsafe or inconsistent measurement")
        returncode = simulation.close(graceful=True)
        simulation = None
        if returncode != 0:
            raise RuntimeError(f"MPI exit {returncode}")
        row.update(safe=True, returncode=returncode)
    except BaseException as error:
        row["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if simulation is not None:
            row["returncode"] = simulation.close()
        if contender is not None:
            contender.close()
            row["workers_reaped"] = all(not process.is_alive()
                                        for process in contender.children)
        row["process_wall_seconds"] = time.monotonic() - started
        (directory / "result.json").write_text(
            json.dumps(row, indent=2, allow_nan=False) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "main"), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--cpu-cores", default="0-7")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    contention_cores = cpu_jitter.parse_cores(args.cpu_cores)
    target_physical_cpus = physical_siblings(contention_cores)
    source = args.checkpoints.resolve() / "manifest.json"
    source_manifest = json.loads(source.read_text())
    records = source_manifest["records"]
    params = Path(source_manifest["initial_restart"]["path"]).parent / "shockparams.mod"
    for record in records:
        if forks.digest(Path(record["restart"])) != record["sha256"]:
            parser.error("checkpoint SHA256 mismatch")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repetitions = 1 if args.phase == "smoke" else args.repetitions
    selected = [records[0], records[-1]] if args.phase == "smoke" else records
    selected_factors = (None, 0.5, 1.0, 1.5) if args.phase == "smoke" else FACTORS
    rng = random.Random(args.seed)
    jobs = []
    for repetition in range(repetitions):
        block = [(record, factor, repetition, regime)
                 for record in selected for factor in selected_factors
                 for regime in ("idle", "contention")]
        rng.shuffle(block)
        jobs.extend(block)

    manifest = {
        "version": "shock-factor-contention-observed-v1",
        "git_commit": base.git_commit(), "mpi_ranks": 32, "omp_threads": 1,
        "skin": 0.5, "probe_steps": PROBE_STEPS, "action_steps": ACTION_STEPS,
        "factors": list(FACTORS), "repetitions": repetitions,
        "random_seed": args.seed, "contention_cores": contention_cores,
        "target_physical_logical_cpus": target_physical_cpus,
        "source_manifest": str(source), "source_manifest_sha256": forks.digest(source),
        "binary_sha256": forks.digest(base.LMP),
        "runner_sha256": forks.digest(Path(__file__)),
        "observation": "common factor-1.0 500-step probe before action",
        "reward": "negative balance application plus following 500-step wall time",
        "schedule": [key(*job) for job in jobs],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("manifest mismatch; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    data = output / "measurements.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines()] if data.exists() else []
    if any(not row["safe"] for row in rows):
        raise RuntimeError("unsafe existing trial requires audit")
    done = {row["key"] for row in rows}
    with data.open("a") as stream:
        for job in jobs:
            identifier = key(*job)
            if identifier in done:
                continue
            row = run_trial(output, job[0], params, *job[1:],
                            contention_cores, target_physical_cpus)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(identifier, "safe", row["safe"],
                  "seconds", row.get("runtime_seconds"), flush=True)
            if not row["safe"]:
                raise RuntimeError(f"unsafe trial {identifier}: {row['errors']}")
            done.add(identifier)
    (output / f"{args.phase}_complete.json").write_text(json.dumps({
        "trials": len(jobs), "completed": len(done), "timestamp": time.time(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
