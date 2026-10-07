#!/usr/bin/env python3
"""Analyze characterization JSONL and write the research report."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

from characterize import ROOT, parse_log


T_CRITICAL = {3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262}


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def cv(values: list[float]) -> float:
    average = mean(values)
    return 100.0 * std(values) / average if average else float("nan")


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def load_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for record in records:
        raw = ROOT / record["raw_directory"]
        parsed = parse_log(raw / "log.lammps")
        if parsed:
            record["lammps"] = parsed
    return records


def load_episode_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def measurement_records(records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if record.get("success") and record.get("scientifically_safe")
        and "state_action_matrix" in record.get("purpose", "")
    ]


def grouped(records: list[dict]) -> dict[tuple[str, str, str, str], list[dict]]:
    result: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        result[
            (
                record["workload"],
                record["backend"],
                record["contention_state"],
                record["action"]["name"],
            )
        ].append(record)
    return result


def values(group: list[dict]) -> list[float]:
    return [float(record["lammps"]["loop_seconds"]) for record in group]


def throughput(group: list[dict]) -> list[float]:
    result = []
    for record in group:
        performance = record["lammps"]
        if "timesteps_per_second" in performance:
            result.append(float(performance["timesteps_per_second"]))
    return result


def matrix_section(records: list[dict]) -> tuple[str, dict]:
    groups = grouped(records)
    report: list[str] = []
    report.append("## State × action performance matrix\n")
    state_summary: dict = {}
    for workload, backend in sorted({(key[0], key[1]) for key in groups}):
        report.append(f"### {workload} / {backend}\n")
        report.append("| State | Action | Mean loop (s) | Std (s) | CV | Throughput | Rank | n |\n")
        report.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
        state_summary[(workload, backend)] = {}
        for state in sorted({key[2] for key in groups if key[:2] == (workload, backend)}):
            state_actions = []
            for (w, b, s, action), group in groups.items():
                if (w, b, s) != (workload, backend, state):
                    continue
                times = values(group)
                rates = throughput(group)
                state_actions.append((mean(times), action, group, times, rates))
            state_actions.sort(key=lambda item: item[0])
            state_summary[(workload, backend)][state] = state_actions
            for rank, (average, action, group, times, rates) in enumerate(state_actions, 1):
                report.append(
                    f"| {state} | `{action}` | {fmt(average)} | {fmt(std(times))} | "
                    f"{fmt(cv(times), 2)}% | {fmt(mean(rates), 2)} step/s | {rank} | {len(times)} |\n"
                )
        report.append("")
    return "\n".join(item.rstrip("\n") for item in report), state_summary


def stability_section(records: list[dict]) -> str:
    report = ["## Measurement stability\n", "| Workload | Backend | Action | Mean loop (s) | Std (s) | CV | n |", "|---|---|---|---:|---:|---:|---:|"]
    for workload, backend in sorted({(r["workload"], r["backend"]) for r in records}):
        group = [
            r for r in records
            if r.get("purpose") == "stability" and r["workload"] == workload and r["backend"] == backend
        ]
        times = values(group)
        if times:
            report.append(f"| {workload} | {backend} | {group[0]['action']['name']} | {fmt(mean(times))} | {fmt(std(times))} | {fmt(cv(times), 2)}% | {len(times)} |")
    report.append("")
    report.append("The measured LAMMPS loop excludes process startup; `process_wall_seconds` is retained in JSONL separately.")
    return "\n".join(report) + "\n"


def interaction_section(state_summary: dict) -> tuple[str, dict]:
    report = ["## Action interaction, relative gaps, and fixed-policy regret\n"]
    conclusions: dict = {}
    for key, states in sorted(state_summary.items()):
        workload, backend = key
        report.append(f"### {workload} / {backend}\n")
        report.append("| State | Best action | Best mean (s) | Runner-up mean (s) | Gap | Gap exceeds conservative CI |\n")
        report.append("|---|---|---:|---:|---:|---|\n")
        state_means: dict[str, dict[str, float]] = {}
        for state, choices in states.items():
            best = choices[0]
            runner = choices[1] if len(choices) > 1 else None
            gap = None if runner is None else (runner[0] - best[0]) / best[0]
            if runner is None:
                separated = False
            else:
                uncertainty = math.sqrt(std(best[3]) ** 2 / len(best[3]) + std(runner[3]) ** 2 / len(runner[3]))
                t_value = T_CRITICAL.get(len(best[3]), 2.0)
                separated = (runner[0] - best[0]) > t_value * uncertainty
            report.append(f"| {state} | `{best[1]}` | {fmt(best[0])} | {fmt(runner[0] if runner else None)} | {fmt(100 * gap if gap is not None else None, 2)}% | {'yes' if separated else 'no'} |")
            state_means[state] = {choice[1]: choice[0] for choice in choices}
        actions = sorted({action for means in state_means.values() for action in means})
        global_means = {action: mean([state_means[state][action] for state in state_means]) for action in actions}
        global_action = min(global_means, key=global_means.get)
        oracle_mean = mean([min(means.values()) for means in state_means.values()])
        penalty = (global_means[global_action] - oracle_mean) / oracle_mean
        best_names = [min(means, key=means.get) for means in state_means.values()]
        changed = len(set(best_names)) > 1
        report.append("")
        report.append("| Fixed-policy statistic | Value |\n|---|---:|\n")
        report.append(f"| Best global fixed action | `{global_action}` ({fmt(global_means[global_action])} s mean across states) |\n")
        report.append(f"| State-dependent oracle mean | {fmt(oracle_mean)} s |\n")
        report.append(f"| Global fixed-policy penalty | {fmt(100 * penalty, 2)}% |\n")
        report.append(f"| Best-action identity changes | {'yes' if changed else 'no'} ({', '.join(best_names)}) |\n")
        report.append("")
        report.append("| State | Action | Regret relative to state oracle |\n|---|---|---:|\n")
        for state, means in state_means.items():
            oracle = min(means.values())
            for action in actions:
                regret = (means[action] - oracle) / oracle
                report.append(f"| {state} | `{action}` | {100 * regret:.2f}% |\n")
        report.append("")
        conclusions[key] = {
            "best_actions": best_names,
            "best_action_changes": changed,
            "global_action": global_action,
            "global_fixed_penalty": penalty,
        }
    return "\n".join(item.rstrip("\n") for item in report), conclusions


def state_observability_section(records: list[dict]) -> str:
    report = ["## Observable state signals\n", "The table uses the first action in each workload/backend to avoid selecting telemetry after seeing the winner.", "", "| Workload | Backend | State | Mean loop (s) | Pair (s) | Neigh (s) | Kspace (s) | Comm (s) | Mean loadavg before | GPU snapshot before |", "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    first_action = {name: info["actions"][0].name for name, info in __import__("characterize").WORKLOADS.items()}
    selected = [r for r in records if r["action"]["name"] == first_action[r["workload"]] and "state_action_matrix" in r.get("purpose", "")]
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in selected:
        groups[(record["workload"], record["backend"], record["contention_state"])].append(record)
    for key in sorted(groups):
        group = groups[key]
        timings = [r["lammps"] for r in group]
        timing_avg = lambda field: mean([t.get("timing_avg_seconds", {}).get(field, float("nan")) for t in timings if field in t.get("timing_avg_seconds", {})])
        loads = [r.get("system_before", {}).get("loadavg_1_5_15", [float("nan")])[0] for r in group]
        gpu = group[0].get("system_before", {}).get("gpu", "").replace("\n", "; ")
        report.append(f"| {key[0]} | {key[1]} | {key[2]} | {fmt(mean([t['loop_seconds'] for t in timings]))} | {fmt(timing_avg('pair'))} | {fmt(timing_avg('neigh'))} | {fmt(timing_avg('kspace'))} | {fmt(timing_avg('comm'))} | {fmt(mean(loads), 2)} | `{gpu[:120]}` |")
    return "\n".join(report) + "\n"


def episode_section(episodes: list[dict], internal_episodes: list[dict] | None = None) -> str:
    report = [
        "## Time-varying episodes\n",
        "The segmented probe keeps one LAMMPS state alive and changes only the external co-runner between 20-step segments. LAMMPS loop time excludes Python/library and co-runner startup; those process-level times remain in the JSONL.",
        "",
        "| Workload | State | n | Mean loop (s) | Std (s) | CV | Pair (s) | Neigh (s) | Kspace (s) | Comm (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in episodes:
        groups[(record["workload"], record["contention_state"])].append(record)
    for (workload, state), group in sorted(groups.items()):
        timings = [r.get("lammps", {}) for r in group]
        times = [float(t["loop_seconds"]) for t in timings]

        def timing(name: str) -> float:
            values = [t.get("timing_avg_seconds", {}).get(name) for t in timings]
            values = [float(value) for value in values if value is not None]
            return mean(values)

        report.append(
            f"| {workload} | {state} | {len(times)} | {fmt(mean(times))} | {fmt(std(times))} | "
            f"{fmt(cv(times), 2)}% | {fmt(timing('pair'))} | {fmt(timing('neigh'))} | "
            f"{fmt(timing('kspace'))} | {fmt(timing('comm'))} |"
        )
    report.extend([
        "",
        "The two orders make contention changes observable in the workload-level timing, but the second SPC/E CPU-contention segment did not slow down as much as the first. This indicates that the simple synthetic co-runner is useful but not perfectly repeatable at this short segment length; static matrix conclusions therefore carry more weight than this probe alone.",
    ])
    if internal_episodes:
        report.extend([
            "",
            "### Idle-only internal-phase probe",
            "",
            "The following is four consecutive idle segments in one LAMMPS process, using the baseline action. Segment 0 is retained to expose warm-up; the post-warm-up CV is calculated over segments 1--3.",
            "",
            "| Workload | n | Segment 0 loop (s) | Segments 1--3 mean (s) | Segments 1--3 CV | Segment 3 loop (s) | Temp 0 → 3 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        by_workload: dict[str, list[dict]] = defaultdict(list)
        for record in internal_episodes:
            by_workload[record["workload"]].append(record)
        for workload, records in sorted(by_workload.items()):
            by_episode: dict[str, list[dict]] = defaultdict(list)
            for record in records:
                by_episode[record["episode_id"]].append(record)
            starts, tails, ends, temp_changes = [], [], [], []
            for group in by_episode.values():
                group.sort(key=lambda record: record["segment_id"])
                if len(group) < 4 or any("loop_seconds" not in r.get("lammps", {}) for r in group):
                    continue
                starts.append(float(group[0]["lammps"]["loop_seconds"]))
                tails.extend(float(r["lammps"]["loop_seconds"]) for r in group[1:])
                ends.append(float(group[-1]["lammps"]["loop_seconds"]))
                temp_changes.append((group[0].get("thermo", {}).get("temp"), group[-1].get("thermo", {}).get("temp")))
            if tails:
                temp_start = mean([pair[0] for pair in temp_changes if pair[0] is not None])
                temp_end = mean([pair[1] for pair in temp_changes if pair[1] is not None])
                report.append(
                    f"| {workload} | {len(starts)} | {fmt(mean(starts))} | {fmt(mean(tails))} | "
                    f"{fmt(cv(tails), 2)}% | {fmt(mean(ends))} | {fmt(temp_start, 2)} → {fmt(temp_end, 2)} |"
                )
        report.extend([
            "",
            "The idle-only probe shows natural temperature evolution, but no sustained monotonic Pair/Kspace change that would by itself establish a new performance regime. The observed LJ first-segment warm-up and the SPC/E final neighbor rebuild are measurement-phase effects, not evidence that the optimal action changes. A longer naturally evolving or spatially imbalanced workload is still needed before treating internal phase as a control state.",
        ])
    return "\n".join(report) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "rl_hpc" / "CHARACTERIZATION.md")
    parser.add_argument("--episodes", type=Path, help="time-varying episode JSONL")
    parser.add_argument("--internal-episodes", type=Path, help="idle-only internal-phase episode JSONL")
    args = parser.parse_args()
    measurement_path = args.measurements if args.measurements.is_absolute() else ROOT / args.measurements
    records = load_records(measurement_path)
    usable = measurement_records(records)
    matrix, state_summary = matrix_section(usable)
    interactions, conclusions = interaction_section(state_summary)
    episodes = None
    internal_episodes = None
    if args.episodes:
        episode_path = args.episodes if args.episodes.is_absolute() else ROOT / args.episodes
        episodes = load_episode_records(episode_path)
    if args.internal_episodes:
        internal_path = args.internal_episodes if args.internal_episodes.is_absolute() else ROOT / args.internal_episodes
        internal_episodes = load_episode_records(internal_path)
    report = [
        "# LAMMPS non-stationarity characterization\n",
        f"Generated {datetime.now(timezone.utc).isoformat()} from `{measurement_path.relative_to(ROOT)}`.",
        "\nThis is a characterization study, not an RL implementation. The analysis uses LAMMPS loop time, not process startup wall time, as the primary performance measure.",
        "\n## Executive result\n",
        "The CPU co-runner produces a clear application state change, especially for the PPPM workload, but the tested CPU neighbor action is nearly state-invariant. The GPU workload is different: same-GPU contention changes the best neighbor configuration, and the fixed-policy penalty is large enough to be practically relevant.",
        "\n" + stability_section(records),
        "## Workloads and experimental scope\n",
        "- Workload A: 128,000-atom homogeneous Lennard-Jones melt, 100 steps per CPU segment and 500 steps per GPU segment.",
        "- Workload B: 36,000-atom SPC/E water, bonded interactions plus PPPM long-range electrostatics, 100 steps per segment.",
        "- CPU: independent `build_rl_char/`, MPI 4, one OpenMP thread/rank.",
        "- GPU: `build_kokkos_cuda/`, one RTX 3090, `gpu/aware off`.",
        "- CPU states: idle, pinned CPU contention, and memory-bandwidth contention. GPU states: idle and same-GPU contention.",
        "- Co-runners: four CPU workers pinned to logical CPUs 0--3; a 512 MiB memcpy stressor pinned to CPUs 8--15; and a CUDA kernel on GPU 0.",
        "- Actions were restricted to safe `skin/every` combinations with `check yes`, `delay 0`, and all measured runs had zero dangerous builds.",
        "\n" + matrix,
        "\n## Static controllability\n",
        "The action response is larger than the idle-run noise for Workload A on both CPU and GPU. For Workload B, the nominal winner changes in some states, but the gaps do not exceed the conservative uncertainty test; this is not yet evidence for state-conditioned CPU action selection.\n",
        interactions,
        "\n" + state_observability_section(usable),
        "## External nonstationarity\n",
        "CPU contention increased the LJ CPU loop time from roughly 1.76--1.82 s to 2.46--2.54 s and the SPC/E CPU loop time from roughly 4.90--4.98 s to 7.60--7.72 s. Memory contention was weaker but measurable (about 5% for LJ and 3--4% for SPC/E). Same-GPU contention increased the LJ GPU loop time by approximately 2.1--4.6× depending on the action.\n",
        episode_section(episodes, internal_episodes) if episodes is not None else "## Time-varying episodes\nNo segmented episode file was supplied to this report generation.\n",
        "## Recommended future environment\n",
        "Retain application timing (`Pair`, `Neigh`, `Kspace`, `Comm`, `Modify`, loop time), neighbor-build counts, backend/resource configuration, and low-overhead load/GPU telemetry. On CPU, discard `skin/every` as a first RL action unless a larger workload or a wider valid range reveals a larger interaction; the tested CPU optimum was effectively constant. On GPU, retain the three neighbor actions and the contention/resource telemetry. Keep MPI rank count, OpenMP count, backend, and GPU mapping as episode-level configuration, not online actions.\n",
        "For an initial fixed-duration environment, use 100 LAMMPS steps per CPU intervention and 500 steps per GPU intervention: these are the stable matrix settings. The 20-step library probe is useful for observing transitions but is too short/noisy for the SPC/E control interval.\n",
        "For the current evidence, online action adaptation is supported for the GPU under external resource contention, while CPU neighbor-action adaptation is not yet justified. The changing episode shows multi-second state changes, so SMDP duration is plausible; however, the short synthetic episode does not yet establish the best variable intervention policy or interval.\n",
        "## Limitations\n",
        "The CPU co-runners are synthetic local stressors, the memory stressor is a simple memcpy loop, and the GPU contender shares GPU 0 only. Measurements were intentionally small and not a production scaling study. The current Open MPI is not CUDA-aware, so `gpu/aware on` was excluded.",
    ]
    args.output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"records_total": len(records), "records_used": len(usable), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
