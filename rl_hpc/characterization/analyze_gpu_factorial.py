#!/usr/bin/env python3
"""Summarize GPU factorial JSONL when a GPU allocation is available."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from characterize import ROOT


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def cv(values: list[float]) -> float:
    return 100.0 * std(values) / mean(values) if values and mean(values) else float("nan")


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def action_label(record: dict) -> str:
    action = record["action"]
    return f"skin={action['skin']}, every={action['neigh_every']}"


def write_report(records: list[dict], output: Path, source: Path) -> None:
    usable = [r for r in records if r.get("success") and r.get("scientifically_safe")]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in usable:
        groups[(record["contention_state"], record["action"]["name"])].append(record)
    report = [
        "# GPU contention factorial analysis",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from `{source.relative_to(ROOT)}`.",
        "",
        "This report does not infer results from failed or unsafe runs. LAMMPS loop time is the primary measure; process wall time and raw GPU telemetry remain in the JSONL.",
        "",
        f"Usable records: {len(usable)} / {len(records)}.",
        "",
        "## State × action matrix",
        "",
        "| State | Skin | Every | Mean runtime (s) | Std (s) | CV | Throughput (step/s) | Neighbor builds | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    state_means: dict[str, dict[str, float]] = defaultdict(dict)
    for (state, action_name), group in sorted(groups.items()):
        runtimes = [float(r["lammps"]["loop_seconds"]) for r in group]
        rates = [float(r["lammps"]["timesteps_per_second"]) for r in group if "timesteps_per_second" in r["lammps"]]
        action = group[0]["action"]
        builds = [float(r["lammps"].get("neighbor_builds", float("nan"))) for r in group]
        report.append(
            f"| {state} | {action['skin']} | {action['neigh_every']} | {fmt(mean(runtimes))} | "
            f"{fmt(std(runtimes))} | {fmt(cv(runtimes), 2)}% | {fmt(mean(rates), 2)} | "
            f"{fmt(mean([x for x in builds if math.isfinite(x)]), 2)} | {len(group)} |"
        )
        state_means[state][action_name] = mean(runtimes)

    report.extend(["", "## Best action and fixed-action regret", ""])
    global_values: dict[str, list[float]] = defaultdict(list)
    oracle_values: list[float] = []
    best_names: dict[str, str] = {}
    for state, actions in sorted(state_means.items()):
        best_action = min(actions, key=actions.get)
        best = actions[best_action]
        best_names[state] = best_action
        oracle_values.append(best)
        report.append(f"### {state}")
        report.append("")
        report.append("| Action | Mean runtime (s) | Regret |")
        report.append("|---|---:|---:|")
        for action_name, runtime in sorted(actions.items(), key=lambda item: item[1]):
            global_values[action_name].append(runtime)
            report.append(f"| `{action_name}` | {fmt(runtime)} | {(runtime - best) / best * 100:.2f}% |")
        report.append("")
        report.append(f"Best action: `{best_action}`; runner-up gap is measured in the matrix above.")
        report.append("")
    if global_values:
        global_action = min(global_values, key=lambda action: mean(global_values[action]))
        global_mean = mean(global_values[global_action])
        oracle_mean = mean(oracle_values)
        report.extend([
            "### Aggregate",
            "",
            f"- Best global fixed action: `{global_action}` ({global_mean:.4f} s mean)",
            f"- State-conditioned oracle mean: {oracle_mean:.4f} s",
            f"- Fixed-policy penalty: {(global_mean - oracle_mean) / oracle_mean * 100:.2f}%",
            f"- Best-action identities: {best_names}",
        ])

    # Marginal means and a simple two-factor interaction residual.
    cell_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in usable:
        action = record["action"]
        cell_values[(record["contention_state"], action["skin"], action["neigh_every"])].append(float(record["lammps"]["loop_seconds"]))
    report.extend(["", "## Factor interpretation", ""])
    for state in sorted({key[0] for key in cell_values}):
        cells = {key[1:]: mean(values) for key, values in cell_values.items() if key[0] == state}
        skin_marginal = {skin: mean([value for (s, _), value in cells.items() if s == skin]) for skin in sorted({key[0] for key in cells})}
        every_marginal = {every: mean([value for (_, e), value in cells.items() if e == every]) for every in sorted({key[1] for key in cells})}
        overall = mean(list(cells.values()))
        residuals = []
        for (skin, every), value in cells.items():
            residuals.append(value - skin_marginal[skin] - every_marginal[every] + overall)
        report.append(f"- `{state}` skin marginal range: {max(skin_marginal.values()) - min(skin_marginal.values()):.4f} s; every marginal range: {max(every_marginal.values()) - min(every_marginal.values()):.4f} s; max interaction residual: {max(abs(x) for x in residuals):.4f} s.")
    output.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "rl_hpc" / "CONTENTION_GPU_ANALYSIS.md")
    args = parser.parse_args()
    source = args.measurements if args.measurements.is_absolute() else ROOT / args.measurements
    output = args.output if args.output.is_absolute() else ROOT / args.output
    write_report(load(source), output, source)
    print(json.dumps({"output": str(output), "source": str(source)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
