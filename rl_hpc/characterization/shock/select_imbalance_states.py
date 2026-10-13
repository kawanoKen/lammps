#!/usr/bin/env python3
"""Select low/medium/high/extreme imbalance states from fixed trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import re

import numpy as np

import select_representative_states as representative


def timing_imbalances(path: Path) -> list[dict[str, float]]:
    """Recover max/average rank timing ratios retained in the LAMMPS log."""
    text = (path.parent / "log.lammps").read_text(errors="replace")
    lines = text.splitlines()
    row_pattern = re.compile(
        r"^\s*(Pair|Neigh|Comm)\s+\|\s+([0-9.eE+-]+)\s+\|\s+"
        r"([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s+\|")
    blocks = []
    for index, line in enumerate(lines):
        match = re.match(r"Loop time of\s+[0-9.eE+-]+\s+on\s+\d+\s+procs\s+for\s+(\d+)\s+steps", line)
        if not match or int(match.group(1)) != 500:
            continue
        end = next((j for j in range(index + 1, len(lines))
                    if lines[j].startswith("Loop time of")), len(lines))
        values = {}
        for candidate in lines[index:end]:
            found = row_pattern.match(candidate)
            if found:
                name = found.group(1).lower()
                average, maximum = float(found.group(3)), float(found.group(4))
                values[f"{name}_rank_imbalance"] = maximum / average
        if len(values) == 3:
            blocks.append(values)
    return blocks


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
    if not 4 <= args.count <= decisions:
        parser.error("count must be between 4 and the trajectory length")

    timing_by_episode = [timing_imbalances(path) for path in paths]
    if any(len(values) != decisions for values in timing_by_episode):
        parser.error("could not align 500-step rank timing blocks with transitions")
    keys = list(representative.features(episodes[0][0])) + [
        "pair_rank_imbalance", "neigh_rank_imbalance", "comm_rank_imbalance"]
    aggregate = []
    for index in range(decisions):
        values = []
        for episode_index, episode in enumerate(episodes):
            value = representative.features(episode[index])
            value.update(timing_by_episode[episode_index][index])
            values.append(value)
        aggregate.append({key: statistics.median(row[key] for row in values)
                          for key in keys})
    matrix = np.asarray([[row[key] for key in keys] for row in aggregate], dtype=float)
    median = np.median(matrix, axis=0)
    scale = np.quantile(matrix, .75, axis=0) - np.quantile(matrix, .25, axis=0)
    scale[scale < 1e-12] = np.std(matrix[:, scale < 1e-12], axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (matrix - median) / scale

    imbalance_keys = ["atom_imbalance", "neighbor_imbalance", "ghost_imbalance",
                      "pair_rank_imbalance", "neigh_rank_imbalance",
                      "comm_rank_imbalance"]
    columns = [keys.index(key) for key in imbalance_keys]
    score = np.mean(normalized[:, columns], axis=1)
    # Anchor the lowest and globally highest load states.  Fill the rest with
    # the highest-load state in successive trajectory windows, preventing a
    # cluster of near-identical extrema from consuming all eight slots.
    selected = [int(np.argmin(score)), int(np.argmax(score))]
    windows = max(1, args.count - 2)
    for block in np.array_split(np.arange(decisions), windows):
        index = int(block[np.argmax(score[block])])
        if index not in selected:
            selected.append(index)
    # Fill duplicate-window slots by farthest-point sampling in observable state.
    while len(selected) < args.count:
        distance = np.min(np.stack([
            np.sum((normalized-normalized[index])**2, axis=1)
            for index in selected]), axis=0)
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    selected.sort()

    q1, q2 = np.quantile(score, [.33, .67])
    payload_rows = []
    for phase, decision in enumerate(selected, 1):
        source = episodes[0][decision]
        tier = "low" if score[decision] <= q1 else ("medium" if score[decision] <= q2 else "high")
        payload_rows.append({
            "phase": phase, "decision": decision,
            "step": int(source["step_start"]),
            "shock_position": statistics.median(
                ep[decision]["state_before"]["shock_position"] for ep in episodes),
            "imbalance_tier": tier,
            "composite_score": float(score[decision]),
            "features": aggregate[decision], "source_episode": 0,
        })
    payload = {
        "method": "imbalance-stratified extrema plus farthest-point coverage",
        "episodes": len(episodes), "decisions_per_episode": decisions,
        "feature_names": keys, "score_features": imbalance_keys,
        "selected": payload_rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("phase decision step tier atom neighbor ghost pairI neighI commI score")
    for row in payload_rows:
        f = row["features"]
        print(row["phase"], row["decision"], row["step"], row["imbalance_tier"],
              f'{f["atom_imbalance"]:.3f}', f'{f["neighbor_imbalance"]:.3f}',
              f'{f["ghost_imbalance"]:.3f}', f'{f["pair_rank_imbalance"]:.3f}',
              f'{f["neigh_rank_imbalance"]:.3f}', f'{f["comm_rank_imbalance"]:.3f}',
              f'{row["composite_score"]:.3f}')


if __name__ == "__main__":
    main()
