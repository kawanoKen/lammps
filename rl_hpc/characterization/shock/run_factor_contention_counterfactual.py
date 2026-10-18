#!/usr/bin/env python3
"""Guarded smoke-to-main supervisor for the observed factor sweep."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import cpu_host_monitor
import cpu_jitter
import shock_counterfactual as base


ROOT = base.ROOT
SCRIPT = ROOT / "rl_hpc/characterization/shock/fork_balance_contention_observed.py"
CHECKPOINTS = (ROOT / "rl_hpc/characterization/data/"
               "shock_representative_checkpoints_fixed47287_v1")
DATA = ROOT / "rl_hpc/characterization/data"
SMOKE = DATA / "shock_factor_contention_observed_smoke_v1"
MAIN = DATA / "shock_factor_contention_observed_main_v1"
STATUS = DATA / "shock_factor_contention_observed_background_v1/status.json"
LOG = DATA / "shock_factor_contention_observed_background_v1/supervisor.log"


def save(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload["updated"] = time.time()
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")


def run(stage: str, output: Path, phase: str) -> None:
    command = [sys.executable, str(SCRIPT), "--checkpoints", str(CHECKPOINTS),
               "--output", str(output), "--phase", phase,
               "--repetitions", "5", "--seed", "20261018",
               "--cpu-cores", "0-7"]
    save({"status": "running", "stage": stage, "command": command})
    with LOG.open("a") as stream:
        stream.write(f"\n[{time.strftime('%F %T')}] {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit {result.returncode}")


def validate_smoke() -> dict:
    rows = [json.loads(line) for line in (SMOKE / "measurements.jsonl").read_text().splitlines()]
    if len(rows) != 16 or not all(row["safe"] for row in rows):
        raise RuntimeError("smoke is incomplete or unsafe")
    summary = {}
    for regime in ("idle", "contention"):
        selected = [row for row in rows if row["regime"] == regime]
        summary[regime] = {
            "n": len(selected),
            "target_util_mean": statistics.fmean(
                row["pre_action_resource_state"]["target_cpu_util_mean"]
                for row in selected),
            "external_busy_per_second": statistics.fmean(
                row["pre_action_resource_state"]["external_busy_core_seconds_estimate"]
                / row["pre_action_resource_state"]["window_seconds"]
                for row in selected),
            "runtime_mean": statistics.fmean(row["runtime_seconds"] for row in selected),
        }
    util_gap = (summary["contention"]["target_util_mean"]
                - summary["idle"]["target_util_mean"])
    external_gap = (summary["contention"]["external_busy_per_second"]
                    - summary["idle"]["external_busy_per_second"])
    summary["target_util_gap"] = util_gap
    summary["external_busy_gap"] = external_gap
    if util_gap < 10.0 or external_gap < 2.0:
        raise RuntimeError(f"contention telemetry separation too weak: {summary}")
    (SMOKE / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    cores = cpu_jitter.parse_cores("0-7")
    state = {"status": "starting", "stage": "baseline_guard"}
    save(state)
    cpu_host_monitor.wait_for_clean_baseline(
        STATUS.parent / "baseline_guard.json", cores,
        window_seconds=5.0, consecutive=2, target_mean_limit=15.0,
        target_max_limit=50.0, system_mean_limit=15.0, load1_limit=8.0,
        timeout_seconds=1800.0)
    monitor = cpu_host_monitor.HostProcessMonitor(
        STATUS.parent / "host_process_monitor.jsonl", cores, 5.0)
    monitor.start()
    try:
        if not (SMOKE / "smoke_complete.json").exists():
            run("smoke", SMOKE, "smoke")
        smoke = validate_smoke()
        save({"status": "running", "stage": "main", "smoke": smoke})
        run("main", MAIN, "main")
        rows = [json.loads(line) for line in (MAIN / "measurements.jsonl").read_text().splitlines()]
        if len(rows) != 800 or not all(row["safe"] for row in rows):
            raise RuntimeError("main collection did not produce 800 safe trials")
        save({"status": "complete", "stage": "complete", "smoke": smoke,
              "main_trials": len(rows)})
    except BaseException as error:
        save({"status": "stopped", "stage": "failed",
              "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        monitor.close()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(
        KeyboardInterrupt("supervisor terminated")))
    main()
