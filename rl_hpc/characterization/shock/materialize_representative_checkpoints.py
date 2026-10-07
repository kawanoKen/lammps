#!/usr/bin/env python3
"""Reproduce the fixed heuristic trajectory and save selected pre-action states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import collect_trajectories as collector
import evaluate_official_balance as official
import online_balance_hybrid as hybrid
import shock_counterfactual as base


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--restart-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-atoms", type=int, default=491520)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    selected = {int(row["decision"]): row for row in selection["selected"]}
    specs = json.loads(args.restart_list.read_text())
    if len(specs) != 1:
        parser.error("exactly one fixed initial state is required")
    spec = specs[0]
    restart = Path(spec["path"]).resolve()
    if not restart.is_file() or digest(restart) != spec["sha256"]:
        parser.error("initial restart is missing or changed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    simulation = collector.Simulation(output)
    records = []
    status = "failed"
    try:
        setup = base.fork_input(restart, hybrid.SKIN, "none", 500, 50).split(
            "variable action_start timer")[0]
        setup += (f"\nneighbor {hybrid.SKIN:g} bin\nrun 0 post no\n"
                  "balance 1.0 shift x 10 1.0 weight neigh 1.5 "
                  "out initial.mesh\nrun 50\n" + collector.OBS)
        state = collector.observation(simulation.execute(setup), hybrid.SKIN, 1.5,
                                      expected_atoms=args.expected_atoms)
        simulation.execute(official.fix_command("official_neigh_10") + "\n")

        final_decision = max(selected)
        for decision in range(final_decision + 1):
            if decision in selected:
                expected_step = int(selected[decision]["step"])
                actual_step = int(state["step"])
                if actual_step != expected_step:
                    raise RuntimeError(
                        f"decision {decision}: step {actual_step} != {expected_step}")
                path = output / f"phase-{selected[decision]['phase']:02d}-step-{actual_step}.restart"
                simulation.execute(f"write_restart {path}\n")
                records.append({
                    **selected[decision],
                    "restart": str(path),
                    "sha256": digest(path),
                    "observed_state": official.compact_state(state),
                })
                print("SAVED", decision, actual_step, path.name, flush=True)

            text = simulation.execute(official.official_commands())
            next_state = collector.observation(text, hybrid.SKIN, state["factor"],
                                               expected_atoms=args.expected_atoms)
            if int(next_state["step"]) != int(state["step"]) + 500:
                raise RuntimeError("non-500-step transition")
            if next_state["previous_segment"]["dangerous_builds"]:
                raise RuntimeError("dangerous neighbor builds")
            state = next_state

        exit_code = simulation.close(graceful=True)
        if exit_code:
            raise RuntimeError(f"MPI exit {exit_code}")
        manifest = {
            "version": "shock-representative-checkpoints-v1",
            "timestamp": time.time(),
            "git_commit": base.git_commit(),
            "initial_restart": spec,
            "selection_file": str(args.selection.resolve()),
            "selection_sha256": digest(args.selection),
            "policy": official.fix_command("official_neigh_10"),
            "initial_partition": "weight neigh 1.5; 50 warmup steps",
            "mpi_ranks": 32,
            "omp_threads": 1,
            "records": records,
            "dangerous_builds": 0,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        status = "complete"
    finally:
        simulation.close()
        (output / "status.json").write_text(json.dumps({
            "status": status,
            "saved": len(records),
            "timestamp": time.time(),
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
