#!/usr/bin/env python3
"""Analyze the representative-state balance action matrix."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def action_name(row: dict) -> str:
    factor = row["action"]["factor"]
    return "skip" if factor is None else f"factor-{factor:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (args.input / "measurements.jsonl").read_text().splitlines()]
    if not rows or any(not row["safe"] for row in rows):
        raise RuntimeError("dataset is empty or contains unsafe trials")
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["phase"]), action_name(row))].append(row["runtime_seconds"])
    phases = sorted({phase for phase, _ in grouped})
    actions = sorted({action for _, action in grouped},
                     key=lambda value: (-1 if value == "skip" else float(value.split("-")[1])))
    matrix, best = [], []
    for phase in phases:
        phase_rows = []
        for action in actions:
            values = grouped[(phase, action)]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            phase_rows.append({"action": action, "n": len(values), "mean": mean,
                               "std": std, "cv": std / mean if mean else 0.0})
        phase_rows.sort(key=lambda row: row["mean"])
        gap = ((phase_rows[1]["mean"] - phase_rows[0]["mean"]) /
               phase_rows[0]["mean"] if len(phase_rows) > 1 else 0.0)
        best.append({"phase": phase, "best": phase_rows[0],
                     "runner_up": phase_rows[1], "relative_gap": gap})
        matrix.append({"phase": phase,
                       "actions": {row["action"]: row for row in phase_rows}})
    totals = {action: sum(next(item for item in matrix if item["phase"] == phase)
                                 ["actions"][action]["mean"] for phase in phases)
              for action in actions}
    fixed = min(totals, key=totals.get)
    oracle = sum(row["best"]["mean"] for row in best)
    result = {"trials": len(rows), "all_safe": True, "phases": phases,
              "actions": actions, "matrix": matrix, "best_by_phase": best,
              "best_fixed_action": fixed, "best_fixed_total": totals[fixed],
              "descriptive_oracle_total": oracle,
              "descriptive_oracle_improvement": (totals[fixed]-oracle)/totals[fixed]}
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Representative state × action analysis", "",
             f"Trials: {len(rows)}; all safe: yes.", "",
             "| Phase | Best | Mean (s) | Std (s) | CV | Runner-up | Gap |",
             "|---:|:---|---:|---:|---:|:---|---:|"]
    for row in best:
        winner = row["best"]
        lines.append(f"| {row['phase']} | {winner['action']} | {winner['mean']:.6f} | "
                     f"{winner['std']:.6f} | {100*winner['cv']:.2f}% | "
                     f"{row['runner_up']['action']} | {100*row['relative_gap']:.2f}% |")
    lines += ["", f"Best fixed action: `{fixed}` ({totals[fixed]:.6f} s summed means).",
              f"Descriptive state oracle: {oracle:.6f} s; improvement "
              f"{100*result['descriptive_oracle_improvement']:.2f}%.", "",
              "The oracle is selected and evaluated on the same samples; validate strong "
              "reversals independently before making a policy-performance claim."]
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
