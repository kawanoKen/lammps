#!/usr/bin/env python3
"""Select representative states from repeated fixed-policy shock trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def features(row: dict) -> dict[str, float]:
    state = row["state_before"]
    segment = state["previous_segment"]
    timing = segment["timing_avg_seconds"]
    steps = max(float(segment["steps"]), 1.0)
    nlocal = segment["nlocal"]
    neighs = segment["neighs"]
    nghost = segment["nghost"]
    return {
        "temperature": float(state["temperature"]),
        "pressure": float(state["pressure"]),
        "atom_imbalance": float(state["atom_imbalance"]),
        "nlocal_range_ratio": (nlocal["max"] - nlocal["min"]) / nlocal["mean"],
        "neighbor_imbalance": neighs["max"] / neighs["mean"],
        "neighbor_range_ratio": (neighs["max"] - neighs["min"]) / neighs["mean"],
        "ghost_imbalance": nghost["max"] / nghost["mean"],
        "pair_per_step": timing["pair"] / steps,
        "neigh_per_step": timing["neigh"] / steps,
        "comm_per_step": timing["comm"] / steps,
        "modify_per_step": timing["modify"] / steps,
        "neighbor_build_rate": float(segment["neighbor_builds"]) / steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input.glob("seed-*/transitions.jsonl"))
    episodes = [[json.loads(line) for line in path.read_text().splitlines()]
                for path in paths]
    if not episodes or len({len(ep) for ep in episodes}) != 1:
        parser.error("missing or unequal-length trajectories")
    decisions = len(episodes[0])
    if not 1 <= args.count <= decisions:
        parser.error("invalid representative count")

    keys = list(features(episodes[0][0]))
    aggregated = []
    for index in range(decisions):
        rows = [features(ep[index]) for ep in episodes]
        aggregated.append({key: statistics.median(row[key] for row in rows)
                           for key in keys})
    matrix = np.array([[row[key] for key in keys] for row in aggregated])
    median = np.median(matrix, axis=0)
    scale = np.quantile(matrix, .75, axis=0) - np.quantile(matrix, .25, axis=0)
    scale[scale < 1e-12] = np.std(matrix[:, scale < 1e-12], axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (matrix - median) / scale

    selected = []
    boundaries = np.linspace(0, decisions, args.count + 1, dtype=int)
    for phase, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        block = normalized[start:end]
        center = np.median(block, axis=0)
        decision = int(start + int(np.argmin(np.sum((block - center) ** 2, axis=1))))
        source = episodes[0][decision]
        selected.append({
            "phase": phase + 1,
            "decision": decision,
            "step": int(source["step_start"]),
            "shock_position": statistics.median(
                ep[decision]["state_before"]["shock_position"] for ep in episodes),
            "features": aggregated[decision],
            "source_episode": 0,
        })

    payload = {
        "method": "eight equal-occupancy trajectory windows; robust observable-state medoid",
        "distance_excludes": ["step", "decision", "shock_position"],
        "episodes": len(episodes),
        "decisions_per_episode": decisions,
        "feature_names": keys,
        "selected": selected,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("phase decision step xshock atom_imb neigh_imb pair/step comm/step")
    for row in selected:
        f = row["features"]
        print(row["phase"], row["decision"], row["step"],
              f'{row["shock_position"]:.3f}', f'{f["atom_imbalance"]:.3f}',
              f'{f["neighbor_imbalance"]:.3f}', f'{f["pair_per_step"]:.6f}',
              f'{f["comm_per_step"]:.6f}')


if __name__ == "__main__":
    main()
