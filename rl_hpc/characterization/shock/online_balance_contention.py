#!/usr/bin/env python3
"""Online FQI under independent L0/L4 wall-clock CPU contention."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
import online_balance_hybrid as hybrid
import online_learn as old
import shock_counterfactual as base


FEATURE_VERSION = "shock-balance-contention-v1"
DATA_VERSION = "shock-balance-contention-v1"
RESOURCE_SCALES = np.array([100.0, 100.0, 50.0])


def policy_state_vector(state: dict, contention_observation: bool) -> np.ndarray:
    application = hybrid.state_vector(state)
    if not contention_observation:
        return application
    resource = state.get("resource_state")
    if not resource:
        raise RuntimeError("contention-aware state lacks resource telemetry")
    values = np.array([
        resource["target_cpu_util_mean"],
        resource["target_cpu_util_max"],
        resource["target_cpu_util_std"],
    ]) / RESOURCE_SCALES
    return np.concatenate((application, values))


def features(state: dict, factor: float | None,
             contention_observation: bool) -> np.ndarray:
    vector = np.clip(policy_state_vector(state, contention_observation), -5, 5)
    execute = float(factor is not None)
    normalized = ((factor - 1.0) / .5) if factor is not None else 0.0
    changed = float(factor is not None and
                    (state["factor"] is None or abs(factor - state["factor"]) > 1e-8))
    action = np.array([execute, normalized, normalized * normalized, changed])
    return np.concatenate((vector, action, vector[1:] * execute,
                           vector[1:] * normalized))


class ContentionFQI(old.OnlineFQI):
    def __init__(self, horizon: int, gamma: float, seed: int,
                 contention_observation: bool):
        super().__init__(horizon, gamma=gamma, seed=seed)
        self.contention_observation = contention_observation

    def add_transition(self, row: dict) -> None:
        self.replay.append(row)
        self.phi.append(features(row["state"], row["factor"],
                                 self.contention_observation))
        candidates = [None] + [float(value) for value in hybrid.BACKUP_FACTORS]
        self.next_candidates.append(np.stack([
            features(row["next_state"], factor, self.contention_observation)
            for factor in candidates]))

    def q(self, state: dict, factor: float | None) -> float:
        if self.coef is None:
            return 0.0
        value = features(state, factor, self.contention_observation) @ self.coef
        return float(np.clip(value, -50.0, 0.0))

    def choose(self, state: dict) -> float | None:
        if self.coef is None:
            return None
        current = state["factor"]
        points: list[float | None] = [None, hybrid.FACTOR_MIN, hybrid.FACTOR_MAX]
        if current is not None and hybrid.FACTOR_MIN <= current <= hybrid.FACTOR_MAX:
            points.append(float(current))
            intervals = ((hybrid.FACTOR_MIN, current),
                         (current, hybrid.FACTOR_MAX))
        else:
            intervals = ((hybrid.FACTOR_MIN, hybrid.FACTOR_MAX),)
        for low, high in intervals:
            if high - low < 1e-5:
                continue
            span = high - low
            xs = np.array([low + .0001 * span, (low + high) / 2,
                           high - .0001 * span])
            ys = np.array([self.q(state, float(value)) for value in xs])
            quadratic, linear, _constant = np.polyfit(xs, ys, 2)
            points.extend(float(value) for value in xs)
            if quadratic < -1e-9:
                vertex = -linear / (2 * quadratic)
                if low < vertex < high:
                    points.append(float(vertex))
        return max(points, key=lambda factor: self.q(state, factor))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_episode(output: Path, episode: int, args: argparse.Namespace,
                restart: Path, physical_seed: int, learner: ContentionFQI,
    cores: tuple[int, ...]) -> dict:
    directory = output / f"episode-{episode:03d}"
    directory.mkdir()
    rng = random.Random(args.seed + episode)
    schedule_seed = (args.seed + episode) ^ 0x5EEDCAFE
    contender = None
    host_monitor = None
    simulation = None
    rewards = []
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
        host_monitor = cpu_host_monitor.HostProcessMonitor(
            directory / "host_process_monitor.jsonl", cores,
            interval_seconds=args.host_monitor_interval_seconds)
        host_monitor.start()
        contender = cpu_contention_clock.WallClockContention(
            cores, schedule_seed, args.block_min_seconds,
            args.block_max_seconds, horizon=args.schedule_horizon_seconds)
        observer = cpu_resource_observer.CpuResourceObserver(
            cores, mpi_ranks=32)
        started = time.monotonic()
        contender.start()
        simulation = collector.Simulation(directory)
        cpu_resource_observer.wait_for_mpi_tree(simulation.process.pid)
        if (episode == args.explore_episodes and learner.coef is None
                and learner.replay):
            update = learner.fit()
            with (directory / "model_updates.jsonl").open("a") as stream:
                stream.write(json.dumps(update) + "\n")

        initial_factor = rng.uniform(hybrid.FACTOR_MIN, hybrid.FACTOR_MAX)
        setup = base.fork_input(restart, hybrid.SKIN, "none", 500, 50).split(
            "variable action_start timer")[0]
        setup += (f"\nneighbor {hybrid.SKIN:g} bin\nrun 0 post no\n"
                  f"balance 1.0 shift x 10 1.0 weight neigh {initial_factor:.12g} "
                  "out initial.mesh\nrun 50\n" + collector.OBS)
        telemetry_begin = observer.snapshot(simulation.process.pid)
        setup_begin = time.monotonic()
        text = simulation.execute(setup)
        setup_end = time.monotonic()
        telemetry_end = observer.snapshot(simulation.process.pid)
        state = collector.observation(text, hybrid.SKIN, initial_factor,
                                      expected_atoms=args.expected_atoms)
        state["partition_mesh"] = (directory / "initial.mesh").read_text()
        state["resource_state"] = observer.summarize(telemetry_begin, telemetry_end)
        hybrid.initialize_control_state(state)
        setup_residency = contender.residency(setup_begin, setup_end)
        (directory / "contention_schedule.json").write_text(json.dumps({
            "seed": schedule_seed, "clock": "time.monotonic",
            "epoch": contender.epoch, "block_min_seconds": args.block_min_seconds,
            "block_max_seconds": args.block_max_seconds,
            "trace": contender.trace, "setup_residency": setup_residency,
        }, indent=2) + "\n")

        with (directory / "transitions.jsonl").open("w") as stream:
            for decision in range(args.decisions):
                before = state
                phase = ("exploration" if episode < args.explore_episodes
                         else "online_learning")
                epsilon = (1.0 if phase == "exploration" else max(
                    .05, .25 - .20 * (episode - args.explore_episodes) /
                    max(1, args.episodes - args.explore_episodes - 1)))
                if phase == "exploration":
                    factor = hybrid.exploration_action(rng, decision,
                                                       args.seed + episode)
                    source = "stratified"
                elif learner.coef is None or rng.random() < epsilon:
                    factor = hybrid.exploration_action(
                        rng, decision, args.seed + episode + 99991)
                    source = "epsilon"
                else:
                    factor = learner.choose(before)
                    source = "greedy"

                telemetry_begin = observer.snapshot(simulation.process.pid)
                worker_before = contender.cpu_seconds()
                interval_begin = time.monotonic()
                text = simulation.execute(hybrid.commands(factor, decision))
                interval_end = time.monotonic()
                worker_after = contender.cpu_seconds()
                telemetry_end = observer.snapshot(simulation.process.pid)

                next_factor = factor if factor is not None else before["factor"]
                state = collector.observation(text, hybrid.SKIN, next_factor,
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
                match = re.findall(r"^TIMES (.*)$", text, re.M)
                if not match:
                    raise RuntimeError("missing timer")
                times = [float(value) for value in match[-1].split()]
                if (len(times) != 4 or
                        not all(math.isfinite(value) and value >= 0 for value in times)
                        or abs(times[0] - sum(times[1:])) > 1e-6):
                    raise RuntimeError("invalid timer identity")
                state.update(
                    intervals_since_balance=(0 if factor is not None else
                                             before["intervals_since_balance"] + 1),
                    previous_balance_applied=factor is not None,
                    previous_balance_seconds=times[2],
                    previous_action_imbalance_before=before["atom_imbalance"],
                    previous_action_imbalance_after=state["atom_imbalance"])
                residency = contender.residency(interval_begin, interval_end)
                reward = -times[0]
                row = {
                    "version": DATA_VERSION, "episode": episode,
                    "decision": decision, "timestamp": time.time(),
                    "physical_seed": physical_seed, "phase": phase,
                    "contention_observation": args.contention_observation,
                    "contention_ground_truth": residency,
                    "contention_schedule_seed": schedule_seed,
                    "contention_worker_seconds": max(0.0, worker_after-worker_before),
                    "state": before,
                    "action": {"balance": factor is not None, "factor": factor,
                               "skin": hybrid.SKIN},
                    "action_source": source, "epsilon": epsilon,
                    "predicted_q": (learner.q(before, factor)
                                    if learner.coef is not None else None),
                    "reward": reward, "cumulative_reward": sum(rewards) + reward,
                    "prepare_seconds": times[1], "balance_seconds": times[2],
                    "run_seconds": times[3], "action_wall_seconds": times[0],
                    "roundtrip_seconds": interval_end - interval_begin,
                    "telemetry_overhead_seconds": max(
                        0.0, interval_end - interval_begin - times[0]),
                    "next_state": state,
                    "neighbor_builds": state["previous_segment"]["neighbor_builds"],
                    "dangerous_builds": state["previous_segment"]["dangerous_builds"],
                    "safe": True,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                rewards.append(reward)
                learner.add_transition({"state": before, "factor": factor,
                                        "reward": reward, "next_state": state,
                                        "decision": decision})
                if (phase == "online_learning" and
                        len(learner.replay) % args.update_every == 0):
                    update = learner.fit()
                    with (directory / "model_updates.jsonl").open("a") as updates:
                        updates.write(json.dumps(update) + "\n")
                if decision % 10 == 0 or decision == args.decisions - 1:
                    print("episode", episode, "decision", decision,
                          "reward", round(reward, 3),
                          "cumulative", round(sum(rewards), 3),
                          "factor", factor, "source", source,
                          "level", residency["start_level"], flush=True)

        exit_code = simulation.close(graceful=True)
        if exit_code != 0:
            raise RuntimeError(f"MPI exit {exit_code}")
        status = "complete"
        result = {
            "episode": episode, "decisions": args.decisions,
            "physical_seed": physical_seed, "restart": str(restart),
            "end_step": state["step"], "reward_total": sum(rewards),
            "mean_reward": statistics.mean(rewards),
            "wall_seconds": time.monotonic() - started,
            "model_updates": learner.updates, "process_exit": exit_code,
        }
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException as exc:
        (directory / "failure.json").write_text(json.dumps({
            "error": repr(exc), "episode": episode,
            "decision": locals().get("decision"), "factor": locals().get("factor"),
            "timestamp": time.time(),
            "recovery": "episode excluded; no partial replay",
        }, indent=2) + "\n")
        raise
    finally:
        if simulation is not None:
            simulation.close()
        if contender is not None:
            contender.close()
        if host_monitor is not None:
            host_monitor.close()
        (directory / "status.json").write_text(json.dumps({
            "status": status, "timestamp": time.time()}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart-list", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--explore-episodes", type=int, default=20)
    parser.add_argument("--decisions", type=int, default=120)
    parser.add_argument("--expected-atoms", type=int, default=491520)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--update-every", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=.95)
    parser.add_argument("--cpu-cores", default="0-7")
    parser.add_argument("--block-min-seconds", type=float, default=45.0)
    parser.add_argument("--block-max-seconds", type=float, default=90.0)
    parser.add_argument("--schedule-horizon-seconds", type=float, default=7200.0)
    parser.add_argument("--contention-observation", action="store_true")
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
    if not 1 <= args.decisions <= 120 or not 0 <= args.explore_episodes < args.episodes:
        parser.error("invalid episode/decision count")
    if (not 0 <= args.gamma < 1 or args.update_every < 1 or
            args.block_min_seconds <= 0 or
            args.block_max_seconds < args.block_min_seconds or
            args.baseline_window_seconds <= 0 or
            args.baseline_consecutive < 1 or
            args.baseline_timeout_seconds <= 0 or
            args.host_monitor_interval_seconds <= 0):
        parser.error("invalid learner/contention settings")
    specs = json.loads(args.restart_list.read_text())
    restarts = [(Path(item["path"]).resolve(), item["physical_seed"])
                for item in specs]
    if not restarts or any(not path.is_file() for path, _seed in restarts):
        parser.error("missing/empty restart list")
    cores = cpu_jitter.parse_cores(args.cpu_cores)
    output = args.output.resolve()
    if args.resume:
        if not (output / "manifest.json").is_file():
            parser.error("resume requires manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
    mode = "aware" if args.contention_observation else "blind"
    feature_version = FEATURE_VERSION + "-" + mode
    manifest = {
        "version": DATA_VERSION, "git_commit": base.git_commit(),
        "restarts": [{"path": str(path), "physical_seed": seed,
                      "sha256": digest(path)} for path, seed in restarts],
        "binary": str(base.LMP), "binary_sha256": digest(base.LMP),
        "script_sha256": digest(Path(__file__)),
        "clock_script_sha256": digest(Path(cpu_contention_clock.__file__)),
        "observer_script_sha256": digest(Path(cpu_resource_observer.__file__)),
        "host_monitor_script_sha256": digest(Path(cpu_host_monitor.__file__)),
        "feature_version": feature_version,
        "contention_observation": args.contention_observation,
        "policy_observation_dimension": 22 if args.contention_observation else 19,
        "state_vector_dimension_including_bias": (
            23 if args.contention_observation else 20),
        "excluded_policy_inputs": ["decision", "step", "shock_position",
                                   "L0/L4 label", "schedule phase",
                                   "future contention", "state history"],
        "skin": hybrid.SKIN,
        "factor_bounds": [hybrid.FACTOR_MIN, hybrid.FACTOR_MAX],
        "actions": "skip balance OR continuous weight-neigh factor",
        "neighbor_policy": "every 1 delay 0 check yes",
        "expected_atoms": args.expected_atoms, "mpi_ranks": 32,
        "omp_threads": 1, "steps_per_decision": 500,
        "decisions": args.decisions, "episodes": args.episodes,
        "explore_episodes": args.explore_episodes,
        "exploration": "50% skip; 50% balance across 30 factor strata",
        "seed": args.seed, "gamma": args.gamma, "ridge": 100.0,
        "update_every": args.update_every,
        "contention": {
            "levels": ["L0", "L4"], "clock": "independent monotonic wall clock",
            "cpu_cores": list(cores),
            "block_seconds": [args.block_min_seconds, args.block_max_seconds],
            "schedule_horizon_seconds": args.schedule_horizon_seconds,
        },
        "baseline_guard": {
            "window_seconds": args.baseline_window_seconds,
            "consecutive": args.baseline_consecutive,
            "target_mean_limit": args.baseline_target_mean_limit,
            "target_max_limit": args.baseline_target_max_limit,
            "system_mean_limit": args.baseline_system_mean_limit,
            "load1_limit": args.baseline_load1_limit,
            "timeout_seconds": args.baseline_timeout_seconds,
            "host_monitor_interval_seconds": args.host_monitor_interval_seconds,
        },
        "reward": "negative action-application plus 500-step wall seconds",
    }
    learner = ContentionFQI(args.decisions, args.gamma, args.seed,
                            args.contention_observation)
    if args.resume:
        if json.loads((output / "manifest.json").read_text()) != manifest:
            parser.error("resume code/configuration/binary differs")
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
        results = progress["results"] if progress else []
        for episode in range(len(results)):
            rows = [json.loads(line) for line in
                    (output / f"episode-{episode:03d}" / "transitions.jsonl")
                    .read_text().splitlines()]
            if len(rows) != args.decisions:
                parser.error(f"incomplete committed episode {episode}")
            for row in rows:
                learner.add_transition({"state": row["state"],
                                        "factor": row["action"]["factor"],
                                        "reward": row["reward"],
                                        "next_state": row["next_state"],
                                        "decision": row["decision"]})
        if progress:
            learner.updates = progress["model_updates"]
            learner.loss = progress["last_loss"]
            if learner.updates:
                with np.load(output / "latest_policy.npz", allow_pickle=False) as data:
                    if str(data["feature_version"]) != feature_version:
                        parser.error("policy feature mismatch")
                    learner.coef = np.array(data["coef"], copy=True)
    else:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        results = []

    for episode in range(len(results), args.episodes):
        attempt = output / f"episode-{episode:03d}"
        if attempt.exists():
            preserved = output / f"interrupted-episode-{episode:03d}-{int(time.time())}"
            attempt.rename(preserved)
        restart, physical_seed = restarts[episode % len(restarts)]
        result = run_episode(output, episode, args, restart, physical_seed,
                             learner, cores)
        results.append(result)
        if learner.coef is not None:
            np.savez(output / "latest_policy.npz", coef=learner.coef,
                     feature_version=feature_version, gamma=args.gamma,
                     source_manifest=str(output / "manifest.json"))
        (output / "progress.json").write_text(json.dumps({
            "complete_episodes": len(results),
            "complete_transitions": sum(item["decisions"] for item in results),
            "results": results, "model_updates": learner.updates,
            "last_loss": learner.loss,
        }, indent=2) + "\n")
        print("COMPLETE episode", episode, "reward",
              round(result["reward_total"], 3), flush=True)
    print("ALL COMPLETE", len(results), flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda _signal, _frame: (_ for _ in ()).throw(
        KeyboardInterrupt("contention study interrupted")))
    main()
