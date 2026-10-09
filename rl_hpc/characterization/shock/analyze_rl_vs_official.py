#!/usr/bin/env python3
"""Combine matched Shock/NEMD RL and official-balance evaluations."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics as stats

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "rl_hpc/characterization/data"
OUT = ROOT / "rl_hpc/docs/experiments/2026-10-09_amp2"
SOURCES = (
    DATA / "shock_balance_rl_eval_matched_official_v2/run/results.json",
    DATA / "shock_official_balance_eval_main_v2/results.json",
    DATA / "shock_official_neigh_08_eval_v1/results.json",
)
DIRS = (
    DATA / "shock_balance_rl_eval_matched_official_v2/run",
    DATA / "shock_official_balance_eval_main_v2",
    DATA / "shock_official_neigh_08_eval_v1",
)
ORDER = (
    "idle_trained", "jitter_trained", "official_neigh_08",
    "official_neigh_10", "official_neigh_15", "official_atoms",
    "official_time_10",
)
LABEL = {
    "idle_trained": "RL-idle",
    "jitter_trained": "RL-contention",
    "official_neigh_08": "neigh 0.8",
    "official_neigh_10": "neigh 1.0",
    "official_neigh_15": "neigh 1.5",
    "official_atoms": "atom count",
    "official_time_10": "time 1.0",
}
REGIME = {"idle": "Idle", "jitter": "CPU contention"}
T_CRIT_95_DF3 = 3.182446


def load_rows() -> list[dict]:
    rows = []
    for source in SOURCES:
        rows.extend(json.loads(source.read_text()))
    expected = {(p, r, s, q) for p in ORDER for r in REGIME
                for s in (47287, 57287) for q in (0, 1)}
    actual = {(x["policy"], x["regime"], x["physical_seed"], x["repeat"])
              for x in rows}
    if actual != expected:
        raise RuntimeError(f"unmatched evaluation matrix: missing={expected-actual}, extra={actual-expected}")
    if any(x["exit_code"] or x["dangerous_builds"] for x in rows):
        raise RuntimeError("unsafe or failed evaluation record")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for regime in REGIME:
        means = {}
        for policy in ORDER:
            values = [x["total_seconds"] for x in rows
                      if x["regime"] == regime and x["policy"] == policy]
            mean, sd = stats.mean(values), stats.stdev(values)
            means[policy] = mean
            result.append({
                "regime": regime, "policy": policy, "label": LABEL[policy],
                "n": len(values), "mean_seconds": mean, "sd_seconds": sd,
                "cv_percent": 100 * sd / mean,
                "ci95_half_seconds": T_CRIT_95_DF3 * sd / math.sqrt(len(values)),
            })
        best = min(means.values())
        for item in result:
            if item["regime"] == regime:
                item["regret_percent"] = 100 * (item["mean_seconds"] - best) / best
                item["rank"] = sorted(means, key=means.get).index(item["policy"]) + 1
    return result


def transition_stats() -> list[dict]:
    accum: dict[tuple[str, str], dict] = {}
    for base in DIRS:
        for path in base.glob("seed-*/transitions.jsonl"):
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if not records or records[0]["policy"] not in ORDER:
                continue
            key = records[0]["regime"], records[0]["policy"]
            slot = accum.setdefault(key, {"segments": 0, "balance": 0, "factors": []})
            for row in records:
                slot["segments"] += 1
                action = row["action"]
                if "balance" in action:
                    slot["balance"] += int(action["balance"])
                    if action["factor"] is not None:
                        slot["factors"].append(float(action["factor"]))
                elif row.get("fix_balance"):
                    slot["balance"] += int(row["fix_balance"]["iterations"] > 0)
    result = []
    for (regime, policy), value in accum.items():
        factors = value["factors"]
        result.append({
            "regime": regime, "policy": policy,
            "balance_event_percent": 100 * value["balance"] / value["segments"],
            "selected_factor_mean": stats.mean(factors) if factors else "",
            "selected_factor_sd": stats.stdev(factors) if len(factors) > 1 else "",
            "segments": value["segments"],
        })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], summary: list[dict]) -> None:
    colors = ["#2563eb", "#7c3aed", "#16a34a", "#65a30d", "#ca8a04", "#ea580c", "#dc2626"]
    width, height = 1800, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 17)
    title = ImageFont.truetype(bold_path, 27)
    draw.text((width / 2, 18), "Shock/NEMD matched live evaluation (mean, 95% t-CI, n=4)",
              fill="black", font=title, anchor="ma")
    y_min, y_max = 740.0, 1180.0
    top, bottom = 90, 625
    for panel, regime in enumerate(REGIME):
        left = 90 + panel * 880
        right = left + 790
        items = sorted((x for x in summary if x["regime"] == regime), key=lambda x: x["rank"])
        def sy(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        for tick in range(800, 1200, 100):
            y = sy(tick)
            draw.line((left, y, right, y), fill="#dddddd", width=1)
            draw.text((left - 10, y), str(tick), fill="black", font=small, anchor="rm")
        draw.text(((left + right) / 2, 58), REGIME[regime], fill="black", font=title, anchor="ma")
        slot = (right - left) / len(items)
        for i, item in enumerate(items):
            x = left + slot * (i + .5)
            half = slot * .32
            y = sy(item["mean_seconds"])
            draw.rectangle((x-half, y, x+half, bottom),
                           fill=colors[ORDER.index(item["policy"])], outline="black")
            high = sy(item["mean_seconds"] + item["ci95_half_seconds"])
            low = sy(item["mean_seconds"] - item["ci95_half_seconds"])
            draw.line((x, high, x, low), fill="black", width=3)
            draw.line((x-8, high, x+8, high), fill="black", width=3)
            draw.line((x-8, low, x+8, low), fill="black", width=3)
            values = [x["total_seconds"] for x in rows
                      if x["regime"] == regime and x["policy"] == item["policy"]]
            offsets = (-12, -4, 4, 12)
            for offset, value in zip(offsets, values):
                py = sy(value)
                draw.ellipse((x+offset-4, py-4, x+offset+4, py+4), fill="black")
            draw.text((x, bottom + 12), item["label"], fill="black", font=small, anchor="ma")
            draw.text((x, bottom + 38), f'{item["mean_seconds"]:.1f} s',
                      fill="black", font=small, anchor="ma")
    draw.text((10, 68), "Total wall time [s]", fill="black", font=font, anchor="la")
    image.save(OUT / "rl_vs_official_all_policies.png")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summary = summarize(rows)
    transitions = transition_stats()
    write_csv(OUT / "rl_vs_official_summary.csv", summary)
    write_csv(OUT / "rl_vs_official_action_stats.csv", transitions)
    plot(rows, summary)
    print(json.dumps({"episodes": len(rows), "summary": summary,
                      "action_stats": transitions}, indent=2))


if __name__ == "__main__":
    main()
