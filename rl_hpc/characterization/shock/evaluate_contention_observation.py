#!/usr/bin/env python3
"""Held-out paired live evaluation of frozen contention-aware/blind policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import signal
import statistics
import time

import numpy as np

import collect_trajectories as collector
import cpu_contention_clock
import cpu_host_monitor
import cpu_jitter
import cpu_resource_observer
import online_balance_contention as learned
import online_balance_hybrid as hybrid
import shock_counterfactual as base


POLICIES = ("aware", "blind", "fixed_08", "fixed_10")
TRAINING = {
    "aware": "shock_balance_contention_aware_v1",
    "blind": "shock_balance_contention_blind_v1",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_models() -> tuple[dict, dict]:
    data = base.ROOT / "rl_hpc/characterization/data"
    models, provenance = {}, {}
    for policy, dataset in TRAINING.items():
        folder = data / dataset
        manifest = json.loads((folder / "manifest.json").read_text())
        progress = json.loads((folder / "progress.json").read_text())
        checkpoint = folder / "latest_policy.npz"
        expected_feature = learned.FEATURE_VERSION + "-" + policy
        if (progress["complete_episodes"] != manifest["episodes"]
                or manifest["feature_version"] != expected_feature
                or manifest["binary_sha256"] != digest(base.LMP)
                or manifest["decisions"] != 120):
            raise RuntimeError(f"incomplete/incompatible model: {policy}")
        model = learned.ContentionFQI(
            120, manifest["gamma"], manifest["seed"], policy == "aware")
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["feature_version"]) != expected_feature:
                raise RuntimeError(f"checkpoint feature mismatch: {policy}")
            model.coef = np.array(saved["coef"], copy=True)
        if not np.isfinite(model.coef).all():
            raise RuntimeError(f"nonfinite model: {policy}")
        models[policy] = model
        provenance[policy] = {
            "dataset": dataset, "checkpoint_sha256": digest(checkpoint),
            "manifest_sha256": digest(folder / "manifest.json"),
            "training_physical_seeds": [row["physical_seed"]
                                        for row in manifest["restarts"]],
        }
    return models, provenance


def select(policy: str, state: dict, models: dict) -> float | None:
    if policy == "fixed_08":
        return 0.8
    if policy == "fixed_10":
        return 1.0
    return models[policy].choose(state)


def run_episode(root: Path, spec: dict, policy: str, repeat: int,
                schedule_seed: int, models: dict, args: argparse.Namespace,
                cores: tuple[int, ...]) -> dict:
    identifier = f'seed-{spec["physical_seed"]}-r{repeat}-{policy}'
    directory = root / identifier
    directory.mkdir()
    contender = None
    monitor = None
    simulation = None
    rows = []
    status = "failed"
    try:
        cpu_host_monitor.wait_for_clean_baseline(
            directory / "baseline_guard.json", cores,
            window_seconds=args.baseline_window_seconds,
            consecutive=args.baseline_consecutive,
            target_mean_limit=args.baseline_target_mean_limit,
            target_max_limit=args.baseline_target_max_limit,
            system_mean_limit=args.baseline_system_mean_limit,
            load1_limit=args.baseline_load1_limit,
            timeout_seconds=args.baseline_timeout_seconds)
        monitor = cpu_host_monitor.HostProcessMonitor(
            directory / "host_process_monitor.jsonl", cores,
            args.host_monitor_interval_seconds)
        monitor.start()
        contender = cpu_contention_clock.WallClockContention(
            cores, schedule_seed, args.block_min_seconds,
            args.block_max_seconds, args.schedule_horizon_seconds)
        contender.start()
        observer = cpu_resource_observer.CpuResourceObserver(cores, mpi_ranks=32)
        started = time.monotonic()
        simulation = collector.Simulation(directory)
        cpu_resource_observer.wait_for_mpi_tree(simulation.process.pid)
        restart = Path(spec["path"]).resolve()
        setup = base.fork_input(restart, hybrid.SKIN, "none", 500, 50).split(
            "variable action_start timer")[0]
        setup += (f"\nneighbor {hybrid.SKIN:g} bin\nrun 0 post no\n"
                  "balance 1.0 shift x 10 1.0 weight neigh 1.5 "
                  "out initial.mesh\nrun 50\n" + collector.OBS)
        telemetry_begin = observer.snapshot(simulation.process.pid)
        setup_begin = time.monotonic()
        text = simulation.execute(setup)
        setup_end = time.monotonic()
        telemetry_end = observer.snapshot(simulation.process.pid)
        state = collector.observation(text, hybrid.SKIN, 1.5,
                                      expected_atoms=args.expected_atoms)
        state["partition_mesh"] = (directory / "initial.mesh").read_text()
        state["resource_state"] = observer.summarize(
            telemetry_begin, telemetry_end)
        hybrid.initialize_control_state(state)
        setup_residency = contender.residency(setup_begin, setup_end)
        (directory / "contention_schedule.json").write_text(json.dumps({
            "seed": schedule_seed, "clock": "time.monotonic",
            "epoch": contender.epoch,
            "block_min_seconds": args.block_min_seconds,
            "block_max_seconds": args.block_max_seconds,
            "trace": contender.trace, "setup_residency": setup_residency,
        }, indent=2) + "\n")

        with (directory / "transitions.jsonl").open("w") as stream:
            for decision in range(args.decisions):
                before = state
                factor = select(policy, before, models)
                telemetry_begin = observer.snapshot(simulation.process.pid)
                worker_before = contender.cpu_seconds()
                interval_begin = time.monotonic()
                text = simulation.execute(hybrid.commands(factor, decision))
                interval_end = time.monotonic()
                worker_after = contender.cpu_seconds()
                telemetry_end = observer.snapshot(simulation.process.pid)
                next_factor = factor if factor is not None else before["factor"]
                state = collector.observation(
                    text, hybrid.SKIN, next_factor,
                    expected_atoms=args.expected_atoms)
                if state["step"] != before["step"] + 500:
                    raise RuntimeError("non-500-step transition")
                if state["shock_position"] > .90 * hybrid.old.LX:
                    raise RuntimeError("front-position guard exceeded")
                mesh = directory / f"partition-{decision:03d}.mesh"
                state["partition_mesh"] = (mesh.read_text() if mesh.exists()
                                           else before["partition_mesh"])
                state["resource_state"] = observer.summarize(
                    telemetry_begin, telemetry_end)
                matches = re.findall(r"^TIMES (.*)$", text, re.M)
                if not matches:
                    raise RuntimeError("missing timer")
                times = [float(value) for value in matches[-1].split()]
                if (len(times) != 4
                        or not all(math.isfinite(x) and x >= 0 for x in times)
                        or abs(times[0] - sum(times[1:])) > 1e-6):
                    raise RuntimeError("invalid timer identity")
                dangerous = state["previous_segment"]["dangerous_builds"]
                if dangerous:
                    raise RuntimeError(f"dangerous neighbor builds: {dangerous}")
                state.update(
                    intervals_since_balance=(0 if factor is not None else
                                             before["intervals_since_balance"] + 1),
                    previous_balance_applied=factor is not None,
                    previous_balance_seconds=times[2],
                    previous_action_imbalance_before=before["atom_imbalance"],
                    previous_action_imbalance_after=state["atom_imbalance"])
                residency = contender.residency(interval_begin, interval_end)
                row = {
                    "decision": decision, "policy": policy,
                    "physical_seed": spec["physical_seed"], "repeat": repeat,
                    "state": before,
                    "action": {"balance": factor is not None,
                               "factor": factor, "skin": hybrid.SKIN},
                    "reward": -times[0], "action_wall_seconds": times[0],
                    "prepare_seconds": times[1],
                    "balance_seconds": times[2], "run_seconds": times[3],
                    "next_state": state, "contention_ground_truth": residency,
                    "contention_worker_seconds": max(
                        0.0, worker_after - worker_before),
                    "roundtrip_seconds": interval_end - interval_begin,
                    "neighbor_builds": state["previous_segment"]["neighbor_builds"],
                    "dangerous_builds": dangerous, "timestamp": time.time(),
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if decision % 10 == 0 or decision == args.decisions - 1:
                    print(identifier, decision,
                          round(sum(x["action_wall_seconds"] for x in rows), 3),
                          factor, residency["start_level"], flush=True)
        exit_code = simulation.close(graceful=True)
        if exit_code != 0:
            raise RuntimeError(f"MPI exit {exit_code}")
        status = "complete"
        result = {
            "id": identifier, "policy": policy,
            "physical_seed": spec["physical_seed"], "repeat": repeat,
            "schedule_seed": schedule_seed, "decisions": args.decisions,
            "total_seconds": sum(row["action_wall_seconds"] for row in rows),
            "run_seconds": sum(row["run_seconds"] for row in rows),
            "balance_seconds": sum(row["balance_seconds"] for row in rows),
            "balance_count": sum(row["action"]["balance"] for row in rows),
            "l0_seconds": sum(row["contention_ground_truth"]["l0_seconds"]
                              for row in rows),
            "l4_seconds": sum(row["contention_ground_truth"]["l4_seconds"]
                              for row in rows),
            "elapsed_seconds_including_startup": time.monotonic() - started,
            "end_step": state["step"], "dangerous_builds": 0,
            "exit_code": exit_code,
        }
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException as error:
        (directory / "failure.json").write_text(json.dumps({
            "error": repr(error), "completed_decisions": len(rows),
            "timestamp": time.time(),
        }, indent=2) + "\n")
        raise
    finally:
        if simulation is not None:
            simulation.close()
        if contender is not None:
            contender.close()
        if monitor is not None:
            monitor.close()
        (directory / "status.json").write_text(json.dumps({
            "status": status, "completed_decisions": len(rows),
            "timestamp": time.time(),
        }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart-list", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--policies", nargs="+", choices=POLICIES,
                        default=list(POLICIES))
    parser.add_argument("--expected-atoms", type=int, default=491520)
    parser.add_argument("--cpu-cores", default="0-7")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--block-min-seconds", type=float, default=45.0)
    parser.add_argument("--block-max-seconds", type=float, default=90.0)
    parser.add_argument("--schedule-horizon-seconds", type=float, default=7200.0)
    parser.add_argument("--baseline-window-seconds", type=float, default=5.0)
    parser.add_argument("--baseline-consecutive", type=int, default=2)
    parser.add_argument("--baseline-target-mean-limit", type=float, default=15.0)
    parser.add_argument("--baseline-target-max-limit", type=float, default=50.0)
    parser.add_argument("--baseline-system-mean-limit", type=float, default=15.0)
    parser.add_argument("--baseline-load1-limit", type=float, default=8.0)
    parser.add_argument("--baseline-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--host-monitor-interval-seconds", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.decisions <= 120 or args.repeats < 1
            or args.repeat_start < 0):
        parser.error("invalid decisions/repeats")
    signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(
        KeyboardInterrupt("evaluation interrupted")))
    models, provenance = frozen_models()
    specs = json.loads(args.restart_list.read_text())
    training_seeds = {seed for row in provenance.values()
                      for seed in row["training_physical_seeds"]}
    for spec in specs:
        restart = Path(spec["path"]).resolve()
        if spec["physical_seed"] in training_seeds:
            parser.error("held-out restart appeared in training")
        if not restart.is_file() or digest(restart) != spec["sha256"]:
            parser.error(f"missing/changed restart: {restart}")
    cores = cpu_jitter.parse_cores(args.cpu_cores)
    output = args.output.resolve()
    if args.resume:
        if not (output / "manifest.json").is_file():
            parser.error("resume requires manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
    jobs, schedules = [], {}
    for repeat in range(args.repeat_start,
                        args.repeat_start + args.repeats):
        for spec in specs:
            schedule_seed = args.seed + 1009 * repeat + spec["physical_seed"]
            trace = cpu_contention_clock.make_trace(
                schedule_seed, args.block_min_seconds,
                args.block_max_seconds, args.schedule_horizon_seconds)
            key = f'{spec["physical_seed"]}-r{repeat}'
            schedules[key] = {"seed": schedule_seed, "trace": trace}
            order = list(args.policies)
            random.Random(schedule_seed ^ 0xA11CE).shuffle(order)
            jobs.extend((spec, repeat, schedule_seed, policy)
                        for policy in order)
    manifest = {
        "version": "shock-contention-observation-eval-v1",
        "git_commit": base.git_commit(), "lammps_binary": str(base.LMP),
        "lammps_sha256": digest(base.LMP),
        "restart_list": str(args.restart_list.resolve()),
        "heldout_restarts": specs, "frozen_policies": provenance,
        "policies": args.policies, "decisions": args.decisions,
        "steps_per_decision": 500, "repeats": args.repeats,
        "repeat_start": args.repeat_start,
        "expected_atoms": args.expected_atoms, "mpi_ranks": 32,
        "omp_threads": 1, "cpu_cores": list(cores),
        "contention": {"clock": "independent monotonic wall clock",
                       "block_seconds": [args.block_min_seconds,
                                         args.block_max_seconds]},
        "schedules": schedules,
        "run_order": [{"physical_seed": spec["physical_seed"],
                       "repeat": repeat, "policy": policy}
                      for spec, repeat, _seed, policy in jobs],
        "fixed_policies": {"fixed_08": "balance factor 0.8 every decision",
                           "fixed_10": "balance factor 1.0 every decision"},
        "evaluation": "frozen; no updates/exploration; independent process/run",
        "reward": "negative action+500-step wall seconds",
        "script_sha256": digest(Path(__file__)),
    }
    if args.resume:
        if json.loads((output / "manifest.json").read_text()) != manifest:
            parser.error("resume code/config/model mismatch")
        results = (json.loads((output / "results.json").read_text())
                   if (output / "results.json").is_file() else [])
    else:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        results = []
    expected_ids = [f'seed-{spec["physical_seed"]}-r{repeat}-{policy}'
                    for spec, repeat, _seed, policy in jobs]
    if [row["id"] for row in results] != expected_ids[:len(results)]:
        parser.error("completed result prefix differs from manifest")
    for spec, repeat, schedule_seed, policy in jobs[len(results):]:
        attempt = output / f'seed-{spec["physical_seed"]}-r{repeat}-{policy}'
        if attempt.exists():
            preserved = attempt.with_name(
                attempt.name + f"-interrupted-{int(time.time())}")
            attempt.rename(preserved)
        result = run_episode(output, spec, policy, repeat, schedule_seed,
                             models, args, cores)
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("COMPLETE", result["id"], result["total_seconds"], flush=True)
    print("ALL COMPLETE", len(results), flush=True)


if __name__ == "__main__":
    main()
