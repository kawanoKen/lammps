#!/usr/bin/env python3
"""Plot the eight representative Shock/NEMD action-response curves."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont


ACTIONS = ("skip", "0.50", "0.75", "1.00", "1.25", "1.50")
COLORS = (
    "#6b7280", "#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def action_name(row: dict) -> str:
    value = row["action"]["factor"]
    return "skip" if value is None else f"{float(value):.2f}"


def load(input_dir: Path) -> dict[int, dict[str, tuple[float, float]]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    path = input_dir / "measurements.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not row["safe"]:
            raise RuntimeError("Unsafe trial found in input dataset")
        grouped[(int(row["phase"]), action_name(row))].append(
            float(row["runtime_seconds"])
        )
    result: dict[int, dict[str, tuple[float, float]]] = {}
    for phase in range(1, 9):
        result[phase] = {}
        for action in ACTIONS:
            values = grouped[(phase, action)]
            if len(values) != 5:
                raise RuntimeError(f"Expected 5 trials for phase={phase}, action={action}")
            result[phase][action] = (statistics.mean(values), statistics.stdev(values))
    return result


def centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
             text_font: ImageFont.ImageFont, fill: str = "#111827") -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text,
              font=text_font, fill=fill)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load(args.input)

    width, height = 2400, 1320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, phase_font = font(38, True), font(26, True)
    axis_font, tick_font, small_font = font(22), font(19), font(17)

    centered(draw, (width / 2, 20),
             "Shock/NEMD: runtime by phase and balance action (mean +/- SD, n=5)",
             title_font)

    outer_left, outer_right = 105, 45
    outer_top, outer_bottom = 95, 100
    gap_x, gap_y = 45, 70
    panel_w = (width - outer_left - outer_right - 3 * gap_x) / 4
    panel_h = (height - outer_top - outer_bottom - gap_y) / 2
    ymin, ymax = 4.3, 9.1
    yticks = (5.0, 6.0, 7.0, 8.0, 9.0)

    for phase in range(1, 9):
        row, col = divmod(phase - 1, 4)
        left = outer_left + col * (panel_w + gap_x)
        top = outer_top + row * (panel_h + gap_y)
        plot_left, plot_right = left + 60, left + panel_w - 15
        plot_top, plot_bottom = top + 48, top + panel_h - 68

        draw.text((left + 8, top), f"Phase {phase}", font=phase_font, fill="#111827")
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)

        def ypix(value: float) -> float:
            return plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top)

        for value in yticks:
            y = ypix(value)
            draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=2)
            label = f"{value:.0f}"
            box = draw.textbbox((0, 0), label, font=tick_font)
            draw.text((plot_left - 10 - (box[2] - box[0]), y - 10), label,
                      font=tick_font, fill="#4b5563")

        values = data[phase]
        winner = min(ACTIONS, key=lambda action: values[action][0])
        slot = (plot_right - plot_left) / len(ACTIONS)
        bar_w = slot * 0.62
        for index, (action, color) in enumerate(zip(ACTIONS, COLORS)):
            mean, std = values[action]
            x = plot_left + (index + 0.5) * slot
            y = ypix(mean)
            base = ypix(ymin)
            draw.rounded_rectangle((x - bar_w / 2, y, x + bar_w / 2, base),
                                   radius=5, fill=color,
                                   outline="#111827" if action == winner else color,
                                   width=4 if action == winner else 1)
            low, high = ypix(mean - std), ypix(mean + std)
            draw.line((x, high, x, low), fill="#111827", width=3)
            draw.line((x - 9, high, x + 9, high), fill="#111827", width=3)
            draw.line((x - 9, low, x + 9, low), fill="#111827", width=3)
            centered(draw, (x, plot_bottom + 10), action, tick_font)
            centered(draw, (x, max(plot_top + 2, y - 25)), f"{mean:.2f}", small_font)
            if action == winner:
                centered(draw, (x, plot_top + 2), "best", small_font, "#991b1b")

        if col == 0:
            draw.text((left - 88, top + panel_h / 2 + 35), "Runtime (s)",
                      font=axis_font, fill="#111827")
        centered(draw, ((plot_left + plot_right) / 2, plot_bottom + 39),
                 "Action: skip or neighbor-weight factor", axis_font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
