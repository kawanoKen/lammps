#!/usr/bin/env python3
"""Validate and visualize rank-targeted Shock balance trials."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rank_target_balance import ATOMS, P, POLICIES, REGIMES, STEPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("evaluation is not complete")
    rows = [json.loads(line) for line in (source / "trials.jsonl").read_text().splitlines()]
    plan = [tuple(item) for item in manifest["trial_order"]]
    observed = [(r["state"], r["regime"], r["policy"], r["repeat"]) for r in rows]
    if len(rows) != len(plan) or len(set(observed)) != len(rows) or set(observed) != set(plan):
        raise RuntimeError("trial plan mismatch")
    for row in rows:
        if row["mpi_returncode"] or row["dangerous_builds"]:
            raise RuntimeError("unsafe trial")
        if row["step_after"] - row["step_before"] != STEPS:
            raise RuntimeError("step mismatch")
        if any(len(row[key]) != P or sum(row[key]) != ATOMS
               for key in ("initial", "applied", "after")):
            raise RuntimeError("rank atom count mismatch")
        if any(not math.isfinite(row[key]) or row[key] < 0
               for key in ("action_seconds", "md_seconds", "total_seconds")):
            raise RuntimeError("nonfinite or negative timing")
    output.mkdir(parents=True, exist_ok=True)
    states = sorted({r["state"] for r in rows})
    lookup = {(r["state"], r["regime"], r["policy"], r["repeat"]): r for r in rows}
    summary = []
    for state in states:
        for regime in REGIMES:
            for policy in POLICIES:
                subset = [r for r in rows if (r["state"], r["regime"], r["policy"]) ==
                          (state, regime, policy)]
                values = [r["total_seconds"] for r in subset]
                mean = statistics.mean(values)
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                summary.append(dict(state=state, regime=regime, policy=policy, n=len(values),
                                    mean_seconds=mean, sd_seconds=sd, cv=sd / mean,
                                    action_mean_seconds=statistics.mean(r["action_seconds"] for r in subset),
                                    md_mean_seconds=statistics.mean(r["md_seconds"] for r in subset),
                                    controller_mean_seconds=statistics.mean(r["controller_seconds"] for r in subset),
                                    end_to_end_mean_seconds=statistics.mean(r["end_to_end_seconds"] for r in subset),
                                    end_to_end_sd_seconds=(statistics.stdev(r["end_to_end_seconds"] for r in subset)
                                                           if len(subset) > 1 else 0.0)))
    with (output / "runtime_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    with (output / "rank_counts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("state", "regime", "policy", "repeat", "rank", "capacity",
                         "target", "initial", "applied", "after"))
        for row in rows:
            for rank in range(P):
                writer.writerow((row["state"], row["regime"], row["policy"], row["repeat"],
                                 rank, row["capacity"][rank], row["target"][rank],
                                 row["initial"][rank], row["applied"][rank], row["after"][rank]))
    fig, axes = plt.subplots(len(states), len(REGIMES), figsize=(15, 3.6 * len(states)),
                             sharex=True, sharey=True, squeeze=False)
    ranks = np.arange(P)
    for i, state in enumerate(states):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            subset = [r for r in rows if r["state"] == state and r["regime"] == regime
                      and r["policy"] == "target_vector"]
            target = np.mean([r["target"] for r in subset], axis=0)
            applied = np.mean([r["applied"] for r in subset], axis=0)
            after = np.mean([r["after"] for r in subset], axis=0)
            ax.plot(ranks, target, color="black", linewidth=2.2, label="target")
            ax.plot(ranks, applied, color="#008f70", marker="o", markersize=3,
                    label="just after action")
            ax.plot(ranks, after, color="#d95f02", linestyle="--", marker="x",
                    markersize=3, label="after 500 steps")
            ax.axvspan(-0.5, 7.5, color="#f6ca75", alpha=0.25)
            ax.set_title(f"{state} / {regime}")
            ax.set_ylabel("atoms owned by rank")
            ax.set_ylim(7000, 21500)
            ax.grid(alpha=0.2)
            ax.set_xticks(range(0, P, 4))
            if i == len(states) - 1:
                ax.set_xlabel("MPI rank (0–7 share physical cores with contender)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output / "target_tracking_by_rank.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(states), len(REGIMES), figsize=(12, 3.2 * len(states)),
                             sharey=False, squeeze=False)
    for i, state in enumerate(states):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            subset = [s for s in summary if s["state"] == state and s["regime"] == regime]
            means = [next(s["mean_seconds"] for s in subset if s["policy"] == p) for p in POLICIES]
            sds = [next(s["sd_seconds"] for s in subset if s["policy"] == p) for p in POLICIES]
            ax.bar(POLICIES, means, yerr=sds, capsize=4,
                   color=("#666666", "#0072b2", "#009e73"))
            ax.set_title(f"{state} / {regime}")
            ax.set_ylabel("action + 500-step runtime (s)")
            ax.tick_params(axis="x", rotation=12)
            ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "runtime_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(states), len(REGIMES), figsize=(12, 3.2 * len(states)),
                             squeeze=False)
    for i, state in enumerate(states):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            subset = [s for s in summary if s["state"] == state and s["regime"] == regime]
            means = [next(s["end_to_end_mean_seconds"] for s in subset if s["policy"] == p)
                     for p in POLICIES]
            sds = [next(s["end_to_end_sd_seconds"] for s in subset if s["policy"] == p)
                   for p in POLICIES]
            ax.bar(POLICIES, means, yerr=sds, capsize=4,
                   color=("#666666", "#0072b2", "#009e73"))
            ax.set_title(f"{state} / {regime}")
            ax.set_ylabel("control + action + 500-step runtime (s)")
            ax.tick_params(axis="x", rotation=12)
            ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "runtime_including_controller.png", dpi=180)
    plt.close(fig)
    tracking = []
    for row in rows:
        if row["policy"] != "target_vector":
            continue
        target = np.asarray(row["target"])
        for phase in ("applied", "after"):
            actual = np.asarray(row[phase])
            tracking.append(dict(state=row["state"], regime=row["regime"],
                                 repeat=row["repeat"], phase=phase,
                                 max_relative_error=float(np.max(np.abs(actual - target) / target)),
                                 mean_absolute_error_atoms=float(np.mean(np.abs(actual - target)))))
    paired = []
    for state in states:
        for regime in REGIMES:
            for baseline in ("skip", "equal_shift"):
                diffs = []
                for repeat in sorted({r["repeat"] for r in rows}):
                    key = (state, regime, baseline, repeat)
                    target_key = (state, regime, "target_vector", repeat)
                    if key in lookup and target_key in lookup:
                        diffs.append(lookup[key]["total_seconds"] -
                                     lookup[target_key]["total_seconds"])
                paired.append(dict(state=state, regime=regime, baseline=baseline,
                                   n=len(diffs), mean_saved_seconds=statistics.mean(diffs),
                                   paired_differences=diffs,
                                   controller_inclusive_mean_saved_seconds=statistics.mean(
                                       lookup[(state, regime, baseline, repeat)]["end_to_end_seconds"] -
                                       lookup[(state, regime, "target_vector", repeat)]["end_to_end_seconds"]
                                       for repeat in sorted({r["repeat"] for r in rows}))))
    (output / "metrics.json").write_text(json.dumps({
        "trials": len(rows), "unsafe_trials": 0, "tracking": tracking, "paired": paired,
        "capacity": manifest["capacity"], "restart_sha256": manifest["restart_sha256"]
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
