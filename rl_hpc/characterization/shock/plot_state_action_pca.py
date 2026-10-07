#!/usr/bin/env python3
"""PCA plots of action-conditioned next software states."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FEATURES = (
    "atom imbalance", "neighbor imbalance", "ghost imbalance",
    "Pair time / step", "Neigh time / step", "Comm time / step",
)
ACTIONS = ("skip", "0.50", "0.75", "1.00", "1.25", "1.50")
COLORS = {
    "skip": (50, 50, 50), "0.50": (0, 114, 178), "0.75": (0, 158, 115),
    "1.00": (230, 159, 0), "1.25": (213, 94, 0), "1.50": (204, 121, 167),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def action_name(row: dict) -> str:
    value = row["action"]["factor"]
    return "skip" if value is None else f"{value:.2f}"


def from_block(block: dict) -> np.ndarray:
    steps = float(block["steps"])
    timing = block["timing_avg_seconds"]
    return np.array([
        block["nlocal"]["max"] / block["nlocal"]["mean"],
        block["neighs"]["max"] / block["neighs"]["mean"],
        block["nghost"]["max"] / block["nghost"]["mean"],
        timing["pair"] / steps, timing["neigh"] / steps,
        timing["comm"] / steps,
    ], dtype=float)


def source_vector(row: dict) -> np.ndarray:
    return from_block(row["pre_state_source"]["previous_segment"])


def transform(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    scale = data.std(axis=0, ddof=0)
    scale[scale < 1e-15] = 1.0
    standardized = (data - mean) / scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    scores = standardized @ vt[:2].T
    variance = singular ** 2 / (len(data) - 1)
    ratio = variance / variance.sum()
    return scores, vt[:2], ratio[:2], np.stack((mean, scale))


def padded_range(values: np.ndarray) -> tuple[float, float]:
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * .09, .15)
    return lo - pad, hi + pad


def projector(bounds, box):
    xmin, xmax, ymin, ymax = bounds
    left, top, right, bottom = box
    def project(point):
        x = left + (point[0] - xmin) / (xmax - xmin) * (right - left)
        y = bottom - (point[1] - ymin) / (ymax - ymin) * (bottom - top)
        return x, y
    return project


def arrow(draw, start, end, color, width=3):
    draw.line((start, end), fill=color, width=width)
    delta = np.array(start) - np.array(end)
    norm = np.linalg.norm(delta)
    if norm < 1e-9:
        return
    unit = delta / norm
    perp = np.array([-unit[1], unit[0]])
    tip = np.array(end)
    p1 = tip + unit * 13 + perp * 6
    p2 = tip + unit * 13 - perp * 6
    draw.polygon([tuple(tip), tuple(p1), tuple(p2)], fill=color)


def diamond(draw, point, radius=8):
    x, y = point
    draw.polygon([(x, y-radius), (x+radius, y), (x, y+radius), (x-radius, y)],
                 fill=(20, 20, 20), outline=(255, 255, 255))


def circle(draw, point, radius, fill, outline=None, width=1):
    x, y = point
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill,
                 outline=outline, width=width)


def draw_legend(draw, x, y):
    draw.text((x, y), "Action", font=font(25, True), fill=(20, 20, 20))
    x += 105
    for action in ACTIONS:
        circle(draw, (x, y+14), 7, COLORS[action])
        label = action if action == "skip" else f"f={action}"
        draw.text((x+12, y), label, font=font(22), fill=(20, 20, 20))
        x += 105 if action == "skip" else 125
    diamond(draw, (x+4, y+14), 7)
    draw.text((x+18, y), "reference s_t", font=font(22), fill=(20, 20, 20))


def axes(draw, box, bounds, xlabel, ylabel):
    left, top, right, bottom = box
    draw.rectangle(box, outline=(120, 120, 120), width=2)
    xmin, xmax, ymin, ymax = bounds
    for i in range(5):
        tx = left + i * (right-left) / 4
        ty = bottom - i * (bottom-top) / 4
        xv = xmin + i * (xmax-xmin) / 4
        yv = ymin + i * (ymax-ymin) / 4
        draw.line((tx, top, tx, bottom), fill=(225, 225, 225), width=1)
        draw.line((left, ty, right, ty), fill=(225, 225, 225), width=1)
        draw.text((tx-25, bottom+8), f"{xv:.1f}", font=font(16), fill=(70,70,70))
        draw.text((left-53, ty-10), f"{yv:.1f}", font=font(16), fill=(70,70,70))
    draw.text(((left+right)//2-70, bottom+34), xlabel, font=font(19), fill=(30,30,30))
    draw.text((left-58, top-30), ylabel, font=font(19), fill=(30,30,30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (args.input / "measurements.jsonl").read_text().splitlines()]
    if len(rows) != 240 or any(not row["safe"] for row in rows):
        raise RuntimeError("expected 240 safe transitions")

    sources = {}
    next_vectors = []
    next_keys = []
    for row in rows:
        phase = int(row["phase"])
        sources.setdefault(phase, source_vector(row))
        next_vectors.append(from_block(row["next_state"]["lammps"]))
        next_keys.append((phase, action_name(row)))
    ordered_sources = np.array([sources[phase] for phase in sorted(sources)])
    all_data = np.vstack((ordered_sources, np.array(next_vectors)))
    scores, components, explained, normalization = transform(all_data)
    source_scores = {phase: scores[index] for index, phase in enumerate(sorted(sources))}
    next_scores = scores[len(sources):]
    grouped = defaultdict(list)
    for key, point in zip(next_keys, next_scores):
        grouped[key].append(point)
    means = {key: np.mean(points, axis=0) for key, points in grouped.items()}
    bounds = (*padded_range(scores[:, 0]), *padded_range(scores[:, 1]))
    bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
    xlabel = f"PC1 ({100*explained[0]:.1f}%)"
    ylabel = f"PC2 ({100*explained[1]:.1f}%)"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (2200, 1450), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 32), "Action-conditioned next software states (global PCA)",
              font=font(37, True), fill=(20,20,20))
    draw.text((90, 83), "Rank partition, neighbor load, and Pair/Neigh/Comm timing; five repetitions per endpoint",
              font=font(23), fill=(70,70,70))
    box = (145, 175, 2110, 1325)
    axes(draw, box, bounds, xlabel, ylabel)
    project = projector(bounds, box)
    for (phase, action), points in grouped.items():
        for point in points:
            circle(draw, project(point), 5, COLORS[action])
    for phase in sorted(sources):
        start = project(source_scores[phase])
        diamond(draw, start, 10)
        draw.text((start[0]+11, start[1]-24), f"S{phase}", font=font(18, True), fill=(20,20,20))
        for action in ACTIONS:
            arrow(draw, start, project(means[(phase, action)]), COLORS[action], 2)
            circle(draw, project(means[(phase, action)]), 9, COLORS[action], (255,255,255), 2)
    draw_legend(draw, 185, 1370)
    canvas.save(args.output_dir / "state_action_pca_global.png")

    facets = Image.new("RGB", (2400, 1560), "white")
    draw = ImageDraw.Draw(facets)
    draw.text((80, 25), "Same-state action forks: PCA of next software state",
              font=font(38, True), fill=(20,20,20))
    draw.text((80, 75), "Global PCA coordinates are shared by all panels; diamond is the heuristic-trajectory reference state",
              font=font(23), fill=(70,70,70))
    panel_w, panel_h = 555, 620
    for idx, phase in enumerate(range(1, 9)):
        col, row_index = idx % 4, idx // 4
        ox, oy = 80 + col * 580, 140 + row_index * 660
        panel_box = (ox+75, oy+55, ox+panel_w-15, oy+panel_h-65)
        draw.text((ox+8, oy+5), f"State S{phase}", font=font(27, True), fill=(20,20,20))
        axes(draw, panel_box, bounds, xlabel, ylabel)
        project = projector(bounds, panel_box)
        start = project(source_scores[phase])
        diamond(draw, start, 9)
        for action in ACTIONS:
            for point in grouped[(phase, action)]:
                circle(draw, project(point), 5, COLORS[action])
            endpoint = project(means[(phase, action)])
            arrow(draw, start, endpoint, COLORS[action], 2)
            circle(draw, endpoint, 8, COLORS[action], (255,255,255), 2)
    draw_legend(draw, 160, 1498)
    facets.save(args.output_dir / "state_action_pca_by_state.png")

    metadata = {
        "features": list(FEATURES),
        "explained_variance_ratio": explained.tolist(),
        "components": {f"PC{index+1}": dict(zip(FEATURES, component.tolist()))
                       for index, component in enumerate(components)},
        "normalization_mean": dict(zip(FEATURES, normalization[0].tolist())),
        "normalization_scale": dict(zip(FEATURES, normalization[1].tolist())),
        "source_note": "s_t is the observed heuristic-trajectory reference; exact reconstructed pre-action rank statistics were not logged",
    }
    (args.output_dir / "state_action_pca_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
