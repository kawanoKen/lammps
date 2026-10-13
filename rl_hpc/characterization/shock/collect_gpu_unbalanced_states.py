#!/usr/bin/env python3
"""Collect a GPU-native Shock trajectory without dynamic load balancing.

The initial partition is made reproducible with one atom-count balance.  No
balance operation is performed after the 50-step warm-up, so physical shock
evolution is allowed to create large rank/GPU load imbalance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import collect_trajectories as collector
import evaluate_official_balance as official
import fork_gpu_state_action as gpu
import online_balance_hybrid as hybrid
import shock_counterfactual as base


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(state: dict) -> dict:
    return {key: value for key, value in state.items() if key != "partition_mesh"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart-list", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected-atoms", type=int, default=491520)
    args = parser.parse_args()
    specs = json.loads(args.restart_list.read_text())
    if len(specs) != 1:
        parser.error("exactly one fixed GPU-native initial restart is required")
    spec = specs[0]
    restart = Path(spec["path"]).resolve()
    if not restart.is_file() or digest(restart) != spec["sha256"]:
        parser.error("initial restart is missing or changed")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": "shock-gpu-unbalanced-state-trajectory-v1",
        "git_commit": base.git_commit(),
        "lammps_binary": str(gpu.GPU_LMP),
        "lammps_sha256": digest(gpu.GPU_LMP),
        "initial_restart": spec,
        "mpi_ranks": 32,
        "omp_threads": 1,
        "gpu_ids": list(gpu.GPU_IDS),
        "steps_per_decision": 500,
        "decisions": args.decisions,
        "repeats": args.repeats,
        "initial_partition": "balance 1.0 shift x 10 1.0; then 50 warm-up steps",
        "dynamic_balance": None,
        "neighbor": "skin 0.5; every 1 delay 0 check yes",
        "selection_intent": "retain naturally developing low through extreme imbalance",
        "script_sha256": digest(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summaries = []
    for repeat in range(args.repeats):
        directory = output / f"seed-{spec['physical_seed']}-idle-r{repeat}-no_balance"
        directory.mkdir()
        simulation = gpu.GpuSimulation(directory)
        rows = []
        status = "failed"
        started = time.monotonic()
        try:
            setup = base.fork_input(restart, hybrid.SKIN, "none", 500, 50).split(
                "variable action_start timer")[0]
            setup += (f"\nneighbor {hybrid.SKIN:g} bin\n"
                      "neigh_modify every 1 delay 0 check yes\n"
                      "run 0 post no\n"
                      "balance 1.0 shift x 10 1.0 out initial.mesh\n"
                      "run 50\n" + collector.OBS)
            state = collector.observation(simulation.execute(setup), hybrid.SKIN, 1.0,
                                          expected_atoms=args.expected_atoms)
            with (directory / "transitions.jsonl").open("w") as stream:
                for decision in range(args.decisions):
                    before = state
                    command = ("variable begin timer\nrun 500\nvariable ended timer\n"
                               "print \"TIMES $(v_ended-v_begin:%.9f)\"\n" +
                               collector.OBS + "\n")
                    text = simulation.execute(command)
                    state = collector.observation(text, hybrid.SKIN, 1.0,
                                                  expected_atoms=args.expected_atoms)
                    match = re.findall(r"^TIMES (.*)$", text, re.M)
                    runtime = float(match[-1]) if match else math.nan
                    if not math.isfinite(runtime) or runtime <= 0:
                        raise RuntimeError("missing/nonfinite segment timer")
                    if state["step"] != before["step"] + 500:
                        raise RuntimeError("non-500-step transition")
                    if state["previous_segment"]["dangerous_builds"]:
                        raise RuntimeError("dangerous neighbor builds")
                    row = {
                        "dataset_version": manifest["version"],
                        "repeat": repeat,
                        "decision": decision,
                        "step_start": before["step"],
                        "step_end": state["step"],
                        "state_before": compact(before),
                        "action": {"balance": "skip", "skin": hybrid.SKIN},
                        "runtime": runtime,
                        "reward": -runtime,
                        "state_after": compact(state),
                        "timestamp": time.time(),
                    }
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    if decision % 10 == 0 or decision == args.decisions - 1:
                        seg = state["previous_segment"]
                        print(repeat, decision, int(state["step"]),
                              f"atom={state['atom_imbalance']:.3f}",
                              f"neigh={seg['neighs']['max']/seg['neighs']['mean']:.3f}",
                              f"runtime={runtime:.3f}", flush=True)
            code = simulation.close(graceful=True)
            if code:
                raise RuntimeError(f"MPI exit {code}")
            status = "complete"
            summaries.append({
                "repeat": repeat,
                "decisions": len(rows),
                "end_step": state["step"],
                "total_seconds": sum(row["runtime"] for row in rows),
                "elapsed_seconds": time.monotonic() - started,
            })
        finally:
            simulation.close()
            (directory / "status.json").write_text(json.dumps({
                "status": status, "completed_decisions": len(rows),
                "timestamp": time.time()}, indent=2) + "\n")
    (output / "results.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
