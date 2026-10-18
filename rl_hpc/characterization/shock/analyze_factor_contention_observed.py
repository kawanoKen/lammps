#!/usr/bin/env python3
"""Analyze the matched CPU-contention balance-factor counterfactual sweep."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont


FACTORS = (None, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
REGIMES = ("idle", "contention")
T95_DF4 = 2.7764451051977987


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def save_runtime_plot(path: Path, cells: list[dict]) -> None:
    image = Image.new("RGB", (1800, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 18), "Same checkpoint: idle vs CPU contention",
              fill="black", font=font(28))
    colors = {"idle": "#1f77b4", "contention": "#ff7f0e"}
    for phase in range(1, 9):
        column, row = (phase - 1) % 4, (phase - 1) // 4
        left, top = 55 + column * 435, 75 + row * 445
        right, bottom = left + 390, top + 350
        selected = [item for item in cells if item["phase"] == phase]
        values = [item["mean"] - item["std"] for item in selected]
        values += [item["mean"] + item["std"] for item in selected]
        low, high = min(values), max(values)
        pad = max(0.1, 0.08 * (high - low))
        low, high = low - pad, high + pad
        draw.rectangle((left, top, right, bottom), outline="#888888", width=1)
        draw.text((left + 5, top + 5), f"S{phase}", fill="black", font=font(20))
        for grid in range(5):
            y = bottom - grid * (bottom - top) / 4
            draw.line((left, y, right, y), fill="#e5e5e5", width=1)
            value = low + grid * (high - low) / 4
            draw.text((left + 3, y - 18), f"{value:.1f}", fill="#555555", font=font(13))
        for regime, offset in (("idle", -5), ("contention", 5)):
            points = []
            for index, factor in enumerate(FACTORS):
                item = next(item for item in selected
                            if item["regime"] == regime and item["factor"] == factor)
                x = left + 25 + index * (right - left - 50) / (len(FACTORS) - 1) + offset
                y = bottom - (item["mean"] - low) / (high - low) * (bottom - top)
                y0 = bottom - (item["mean"] - item["std"] - low) / (high - low) * (bottom - top)
                y1 = bottom - (item["mean"] + item["std"] - low) / (high - low) * (bottom - top)
                draw.line((x, y0, x, y1), fill=colors[regime], width=2)
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[regime])
                points.append((x, y))
            draw.line(points, fill=colors[regime], width=2)
        for index, factor in enumerate(FACTORS):
            x = left + 25 + index * (right - left - 50) / (len(FACTORS) - 1)
            draw.text((x - 14, bottom + 5), label(factor), fill="black", font=font(11))
    draw.rectangle((1430, 25, 1450, 42), fill=colors["idle"])
    draw.text((1458, 20), "idle", fill="black", font=font(18))
    draw.rectangle((1530, 25, 1550, 42), fill=colors["contention"])
    draw.text((1558, 20), "L4", fill="black", font=font(18))
    image.save(path)


def save_telemetry_plot(path: Path, telemetry: list[dict]) -> None:
    names = ("cpu_util_mean", "external_busy_cores", "comm_per_step", "pair_per_step")
    image = Image.new("RGB", (1600, 430), "white")
    draw = ImageDraw.Draw(image)
    for index, name in enumerate(names):
        item = next(value for value in telemetry if value["feature"] == name)
        left, top = 35 + index * 395, 60
        right, bottom = left + 350, 350
        means = [item["idle_mean"], item["contention_mean"]]
        stds = [item["idle_std"], item["contention_std"]]
        low = min(mean - std for mean, std in zip(means, stds))
        high = max(mean + std for mean, std in zip(means, stds))
        if high == low:
            high += 1.0
        pad = 0.1 * (high - low)
        low, high = low - pad, high + pad
        draw.rectangle((left, top, right, bottom), outline="#888888")
        draw.text((left, 8), name, fill="black", font=font(18))
        draw.text((left, 32), f"AUC={item['auc']:.3f}", fill="#333333", font=font(16))
        for position, (mean, std, color, title) in enumerate(zip(
                means, stds, ("#1f77b4", "#ff7f0e"), ("idle", "L4"))):
            x = left + 105 + position * 145
            y = bottom - (mean - low) / (high - low) * (bottom - top)
            y0 = bottom - (mean - std - low) / (high - low) * (bottom - top)
            y1 = bottom - (mean + std - low) / (high - low) * (bottom - top)
            draw.line((x, y0, x, y1), fill=color, width=5)
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
            draw.text((x - 18, bottom + 8), title, fill="black", font=font(15))
            draw.text((x - 50, y - 30), f"{mean:.4g}", fill=color, font=font(14))
    image.save(path)


def label(factor: float | None) -> str:
    return "skip" if factor is None else f"{factor:g}"


def mean_ci(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half = T95_DF4 * std / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "std": std,
            "cv_percent": 100.0 * std / abs(mean) if mean else None,
            "ci95": [mean - half, mean + half], "samples": values}


def paired_difference(first: list[float], second: list[float]) -> dict:
    return mean_ci([a - b for a, b in zip(first, second)])


def auc(idle: list[float], contention: list[float]) -> float:
    score = sum((b > a) + 0.5 * (b == a) for a in idle for b in contention)
    raw = score / (len(idle) * len(contention))
    return max(raw, 1.0 - raw)


def scalar_features(row: dict) -> dict[str, float]:
    resource = row["pre_action_resource_state"]
    timing = row["state_before_action"]["previous_segment"]["timing_avg_seconds"]
    return {
        "cpu_util_mean": resource["target_cpu_util_mean"],
        "cpu_util_max": resource["target_cpu_util_max"],
        "cpu_util_std": resource["target_cpu_util_std"],
        "external_busy_cores": (
            resource["external_busy_core_seconds_estimate"] / resource["window_seconds"]),
        "lammps_cpu_entitlement": resource["lammps_cpu_entitlement"],
        "rank_cpu_efficiency_min": resource["rank_cpu_efficiency"]["minimum"],
        "rank_cpu_efficiency_std": resource["rank_cpu_efficiency"]["std"],
        "involuntary_context_switch_rate": (
            resource["rank_involuntary_context_switches_per_second"]["mean"]),
        "pair_per_step": timing["pair"] / 500.0,
        "neigh_per_step": timing["neigh"] / 500.0,
        "comm_per_step": timing["comm"] / 500.0,
    }


def leave_one_phase_out_accuracy(rows: list[dict], feature: str) -> float:
    correct = total = 0
    for held_out in range(1, 9):
        train = [row for row in rows if row["phase"] != held_out]
        test = [row for row in rows if row["phase"] == held_out]
        idle_mean = statistics.fmean(scalar_features(row)[feature]
                                     for row in train if row["regime"] == "idle")
        active_mean = statistics.fmean(scalar_features(row)[feature]
                                       for row in train if row["regime"] == "contention")
        threshold = 0.5 * (idle_mean + active_mean)
        increasing = active_mean > idle_mean
        for row in test:
            value = scalar_features(row)[feature]
            predicted = "contention" if ((value > threshold) == increasing) else "idle"
            correct += predicted == row["regime"]
            total += 1
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (args.input / "measurements.jsonl").read_text().splitlines()]
    if (len(rows) != 800 or not all(row["safe"] for row in rows)
            or any(row["state_before_action"]["previous_segment"]["dangerous_builds"]
                   or row["next_state"]["previous_segment"]["dangerous_builds"]
                   for row in rows)):
        raise RuntimeError("dataset is incomplete or unsafe")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["regime"], row["action"]["factor"])].append(row)
    for selected in groups.values():
        selected.sort(key=lambda row: row["repetition"])
        if [row["repetition"] for row in selected] != list(range(5)):
            raise RuntimeError("missing or duplicated repetition")

    cells = []
    best = {}
    for phase in range(1, 9):
        for regime in REGIMES:
            ranked = []
            for factor in FACTORS:
                runtimes = [row["runtime_seconds"]
                            for row in groups[(phase, regime, factor)]]
                item = {"phase": phase, "regime": regime, "factor": factor,
                        **mean_ci(runtimes)}
                cells.append(item)
                ranked.append(item)
            ranked.sort(key=lambda item: item["mean"])
            winner, runner_up = ranked[:2]
            difference = paired_difference(winner["samples"], runner_up["samples"])
            best[(phase, regime)] = {
                "winner": winner, "runner_up": runner_up,
                "winner_minus_runner_up": difference,
                "resolved": difference["ci95"][1] < 0,
            }

    interactions = []
    for phase in range(1, 9):
        idle_factor = best[(phase, "idle")]["winner"]["factor"]
        active_factor = best[(phase, "contention")]["winner"]["factor"]
        idle_a = [row["runtime_seconds"] for row in groups[(phase, "idle", idle_factor)]]
        idle_b = [row["runtime_seconds"] for row in groups[(phase, "idle", active_factor)]]
        active_a = [row["runtime_seconds"]
                    for row in groups[(phase, "contention", idle_factor)]]
        active_b = [row["runtime_seconds"]
                    for row in groups[(phase, "contention", active_factor)]]
        # Difference-in-differences: (B-A)_contention - (B-A)_idle.
        did = [(b2 - a2) - (b1 - a1)
               for a1, b1, a2, b2 in zip(idle_a, idle_b, active_a, active_b)]
        result = mean_ci(did)
        interactions.append({
            "phase": phase, "idle_best": idle_factor,
            "contention_best": active_factor, **result,
            "resolved_reversal": (idle_factor != active_factor
                                  and result["ci95"][1] < 0),
        })

    fixed = {}
    for regime in REGIMES:
        ranked = []
        for factor in FACTORS:
            total = sum(next(item["mean"] for item in cells
                             if item["phase"] == phase and item["regime"] == regime
                             and item["factor"] == factor)
                        for phase in range(1, 9))
            ranked.append((total, factor))
        ranked.sort()
        oracle = sum(best[(phase, regime)]["winner"]["mean"] for phase in range(1, 9))
        fixed[regime] = {"factor": ranked[0][1], "total": ranked[0][0],
                         "runner_up_factor": ranked[1][1],
                         "runner_up_total": ranked[1][0], "oracle": oracle,
                         "oracle_improvement_percent": 100 * (ranked[0][0] - oracle) / ranked[0][0]}
    globally_ranked = []
    for factor in FACTORS:
        total = sum(next(item["mean"] for item in cells
                         if item["phase"] == phase and item["regime"] == regime
                         and item["factor"] == factor)
                    for phase in range(1, 9) for regime in REGIMES)
        globally_ranked.append((total, factor))
    globally_ranked.sort()
    phase_only = sum(min(
        sum(next(item["mean"] for item in cells
                 if item["phase"] == phase and item["regime"] == regime
                 and item["factor"] == factor) for regime in REGIMES)
        for factor in FACTORS) for phase in range(1, 9))
    full_oracle = sum(best[(phase, regime)]["winner"]["mean"]
                      for phase in range(1, 9) for regime in REGIMES)
    opportunities = {
        "global_fixed_factor": globally_ranked[0][1],
        "global_fixed_total": globally_ranked[0][0],
        "phase_only_oracle": phase_only, "phase_and_regime_oracle": full_oracle,
        "phase_only_gain_vs_fixed_percent": 100 * (globally_ranked[0][0] - phase_only) / globally_ranked[0][0],
        "regime_information_gain_vs_phase_only_percent": 100 * (phase_only - full_oracle) / phase_only,
        "full_oracle_gain_vs_fixed_percent": 100 * (globally_ranked[0][0] - full_oracle) / globally_ranked[0][0],
    }

    feature_names = list(scalar_features(rows[0]))
    telemetry = []
    for feature in feature_names:
        idle = [scalar_features(row)[feature] for row in rows if row["regime"] == "idle"]
        active = [scalar_features(row)[feature] for row in rows
                  if row["regime"] == "contention"]
        telemetry.append({
            "feature": feature, "idle_mean": statistics.fmean(idle),
            "idle_std": statistics.stdev(idle),
            "contention_mean": statistics.fmean(active),
            "contention_std": statistics.stdev(active), "auc": auc(idle, active),
            "leave_one_phase_out_threshold_accuracy":
                leave_one_phase_out_accuracy(rows, feature),
        })

    physical_ranges = {}
    for name in ("temperature", "pressure", "density", "total_energy_per_atom",
                 "shock_position"):
        physical_ranges[name] = max(
            max(row["state_before_action"][name] for row in rows if row["phase"] == phase)
            - min(row["state_before_action"][name] for row in rows if row["phase"] == phase)
            for phase in range(1, 9))

    result = {
        "dataset_sha256": hashlib.sha256(
            (args.input / "measurements.jsonl").read_bytes()).hexdigest(),
        "trials": len(rows), "cells": cells, "best": [
            {"phase": phase, "regime": regime, **value}
            for (phase, regime), value in best.items()],
        "interactions": interactions, "fixed": fixed,
        "opportunities": opportunities, "telemetry": telemetry,
        "safety": {"safe": sum(row["safe"] for row in rows),
                   "dangerous_builds": 0, "physical_prestate_max_ranges": physical_ranges},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")

    with (args.output / "runtime_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "phase", "regime", "factor", "mean", "std", "cv_percent",
            "ci95_low", "ci95_high"))
        writer.writeheader()
        for item in cells:
            writer.writerow({**{name: item[name] for name in
                                ("phase", "regime", "factor", "mean", "std", "cv_percent")},
                             "ci95_low": item["ci95"][0],
                             "ci95_high": item["ci95"][1]})

    save_runtime_plot(args.output / "factor_runtime_by_phase.png", cells)
    save_telemetry_plot(args.output / "telemetry_separation.png", telemetry)

    lines = [
        "# CPU contention × balance factor反実仮想実験：結果", "",
        "同一checkpointからidle/L4および10 actionsへ分岐した800試行の結果。各cellは5反復。",
        "CIは反復blockを対応させたdf=4の無補正95% t区間であり、多重比較の同時保証ではない。", "",
        "## Best action", "",
        "| State | Idle best | gap (95% CI), s | L4 best | gap (95% CI), s | reversal interaction |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for phase in range(1, 9):
        idle = best[(phase, "idle")]
        active = best[(phase, "contention")]
        interaction = interactions[phase - 1]
        def gap_text(item):
            value = item["winner_minus_runner_up"]
            return f"{value['mean']:+.4f} [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]"
        verdict = ("resolved" if interaction["resolved_reversal"] else
                   ("same action" if interaction["idle_best"] == interaction["contention_best"]
                    else "unresolved"))
        lines.append(f"| S{phase} | {label(idle['winner']['factor'])} | {gap_text(idle)} | "
                     f"{label(active['winner']['factor'])} | {gap_text(active)} | {verdict} |")
    lines += ["", "負のgapはwinnerがrunner-upより速いことを表す。`resolved`は、idle winnerとL4 winnerの"
              "difference-in-differences 95% CIが0を跨がない順位反転。", "",
              "![状態別runtime](factor_runtime_by_phase.png)", "",
              "## 固定actionと記述的oracle", "",
              "| Regime | Best Fixed | Fixed total, s | State oracle, s | improvement |",
              "|:---|---:|---:|---:|---:|"]
    for regime in REGIMES:
        item = fixed[regime]
        lines.append(f"| {regime} | {label(item['factor'])} | {item['total']:.4f} | "
                     f"{item['oracle']:.4f} | {item['oracle_improvement_percent']:.2f}% |")
    lines += ["", f"全条件共通Best Fixedはfactor {label(opportunities['global_fixed_factor'])}。"
              f"内部phaseだけを使うoracleの改善は{opportunities['phase_only_gain_vs_fixed_percent']:.2f}%、"
              f"phaseに加えてcontention labelを使う追加改善は"
              f"{opportunities['regime_information_gain_vs_phase_only_percent']:.2f}%、"
              f"完全なphase×regime oracleはBest Fixedより"
              f"{opportunities['full_oracle_gain_vs_fixed_percent']:.2f}%短い。"
              "これらは同じデータで選択・評価した記述的上限であり、学習方策の汎化性能ではない。", "",
              "## Action前telemetry", "",
              "| Feature | Idle mean | L4 mean | AUC | leave-one-phase-out accuracy |",
              "|:---|---:|---:|---:|---:|"]
    for item in telemetry:
        lines.append(f"| {item['feature']} | {item['idle_mean']:.6g} | "
                     f"{item['contention_mean']:.6g} | {item['auc']:.3f} | "
                     f"{100*item['leave_one_phase_out_threshold_accuracy']:.1f}% |")
    lines += ["", "![telemetry分離](telemetry_separation.png)", "",
              "## 安全性と解釈", "",
              "- 800/800試行が成功し、probe/action区間ともdangerous buildは0。",
              "- 同一phase内のaction前temperature、pressure、density、energy、shock positionのrangeは記録精度で0。",
              "- CPU utilization meanと外部busy推定はL0/L4を完全分離した。CPU utilization maxは常に100%で無効。",
              "- Comm timeにも強い識別情報があり、従来のBlind方策が外乱を間接観測できた説明と整合する。",
              "- interaction contrastも同じgridからwinnerを選んだ後に評価しているため探索的であり、選択biasを含む。",
              "- S2は両regimeのwinner-runner gapとinteractionがすべて解決しており、現時点で最も強いreversal。",
              "- S4--S8はinteractionは大きいが各regimeの最小点周辺が平坦な場合があり、exact argminの確定には候補actionの独立再測定が必要。",
              ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
