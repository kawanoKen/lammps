#!/usr/bin/env python3
"""Plot the audited shock online-learning reward trace as a standalone SVG.

Uses only the Python standard library; no learning claim is inferred from it.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def polyline(points: list[tuple[float, float]], color: str, width: float = 2) -> str:
    xy = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return (f'<polyline points="{xy}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" '
            'stroke-linejoin="round"/>')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    episodes = read_csv(root / 'episode_reward_summary.csv')
    trace = read_csv(root / 'reward_trace.csv')
    if len(episodes) != manifest['episodes'] or any(e['complete'] != 'True' for e in episodes):
        raise SystemExit('plot requires a fully audited, complete dataset')
    n = manifest['episodes']
    n_explore = manifest['explore_episodes']
    horizon = manifest['decisions']
    if len(trace) != n * horizon:
        raise SystemExit('reward trace length does not match manifest')
    by_episode: dict[int, list[float]] = {}
    for row in trace:
        by_episode.setdefault(int(row['episode']), []).append(float(row['reward_seconds']))
    if any(len(values) != horizon for values in by_episode.values()):
        raise SystemExit('incomplete episode in reward trace')

    # Match physical restart identities: four final exploration episodes and
    # four final learning episodes each cover the same four training seeds.
    early_ids = list(range(n_explore - 4, n_explore))
    late_ids = list(range(n - 4, n))
    if n_explore < 4 or n - n_explore < 4:
        raise SystemExit('need at least four episodes in both phases')
    def rolling_average(ids: list[int]) -> list[float]:
        curves = []
        for episode in ids:
            values = by_episode[episode]
            curves.append([mean(values[max(0, i - 9):i + 1])
                           for i in range(horizon)])
        return [mean(curve[i] for curve in curves) for i in range(horizon)]
    early = rolling_average(early_ids)
    late = rolling_average(late_ids)
    totals = [float(e['total_reward_seconds']) for e in episodes]

    width, height = 1100, 720
    left, right = 90, 1050
    top1, bottom1 = 78, 345
    top2, bottom2 = 425, 645
    fg, muted, grid = '#192637', '#546579', '#d8e0e8'
    exploratory, learned = '#9a673d', '#156d98'
    def x1(i: float) -> float:
        return left + (right - left) * i / (n - 1)
    ymin1 = (int(min(totals) / 50) - 1) * 50
    ymax1 = (int(max(totals) / 50) + 2) * 50
    def y1(value: float) -> float:
        return bottom1 - (value - ymin1) / (ymax1 - ymin1) * (bottom1 - top1)
    def x2(i: float) -> float:
        return left + (right - left) * i / (horizon - 1)
    ymin2 = int(min(early + late)) - 1
    ymax2 = int(max(early + late)) + 2
    def y2(value: float) -> float:
        return bottom2 - (value - ymin2) / (ymax2 - ymin2) * (bottom2 - top2)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">Shock/NEMD online learning reward progression</title>',
        '<desc id="desc">Top: total negative elapsed-time reward across 36 episodes. '
        'Bottom: ten-decision trailing reward averaged over the final four exploration '
        'and final four learning episodes. These descriptive traces do not establish '
        'a causal learning speedup.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" fill="{fg}" font-family="sans-serif" '
        'font-size="22" font-weight="600">Shock/NEMD reward progression</text>',
        f'<text x="{left}" y="56" fill="{muted}" font-family="sans-serif" '
        'font-size="13">r = -elapsed seconds for action + 500 MD steps; higher is better</text>',
    ]
    for yvalue in range(ymin1, ymax1 + 1, 50):
        y = y1(yvalue)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
                     f'stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                     f'fill="{muted}" font-family="sans-serif" font-size="12">{yvalue}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom1}" x2="{right}" y2="{bottom1}" '
                 f'stroke="{fg}"/>')
    parts.append(f'<line x1="{left}" y1="{top1}" x2="{left}" y2="{bottom1}" '
                 f'stroke="{fg}"/>')
    split_x = (x1(n_explore - 1) + x1(n_explore)) / 2
    parts.append(f'<line x1="{split_x:.2f}" y1="{top1}" x2="{split_x:.2f}" '
                 f'y2="{bottom1}" stroke="{muted}" stroke-dasharray="5 4"/>')
    parts.append(polyline([(x1(i), y1(v)) for i, v in enumerate(totals[:n_explore])], exploratory))
    parts.append(polyline([(x1(i), y1(v)) for i, v in enumerate(totals[n_explore:], n_explore)], learned))
    for i, value in enumerate(totals):
        color = exploratory if i < n_explore else learned
        x, y = x1(i), y1(value)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}">'
                     f'<title>Episode {i}: {value:.1f} s reward</title></circle>')
    for i in (0, 4, 8, 12, 16, 20, 24, 28, 32, 35):
        parts.append(f'<text x="{x1(i):.2f}" y="{bottom1 + 20}" text-anchor="middle" '
                     f'fill="{muted}" font-family="sans-serif" font-size="12">{i}</text>')
    parts.append(f'<text x="{(left + right) / 2}" y="{bottom1 + 40}" text-anchor="middle" '
                 f'fill="{fg}" font-family="sans-serif" font-size="13">Episode</text>')
    parts.append(f'<text x="{left}" y="{top1 - 10}" fill="{fg}" '
                 'font-family="sans-serif" font-size="13">Total reward (s)</text>')
    parts.append(f'<text x="{x1(3):.2f}" y="{top1 + 18}" fill="{exploratory}" '
                 'font-family="sans-serif" font-size="13">Exploration (0–11)</text>')
    parts.append(f'<text x="{x1(15):.2f}" y="{top1 + 18}" fill="{learned}" '
                 'font-family="sans-serif" font-size="13">Online learning (12–35)</text>')

    for yvalue in range(ymin2, ymax2 + 1):
        y = y2(yvalue)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
                     f'stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                     f'fill="{muted}" font-family="sans-serif" font-size="12">{yvalue}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom2}" x2="{right}" y2="{bottom2}" '
                 f'stroke="{fg}"/>')
    parts.append(f'<line x1="{left}" y1="{top2}" x2="{left}" y2="{bottom2}" '
                 f'stroke="{fg}"/>')
    parts.append(polyline([(x2(i), y2(v)) for i, v in enumerate(early)], exploratory, 2.5))
    parts.append(polyline([(x2(i), y2(v)) for i, v in enumerate(late)], learned, 2.5))
    for i in (0, 20, 40, 60, 80, 100, 119):
        parts.append(f'<text x="{x2(i):.2f}" y="{bottom2 + 20}" text-anchor="middle" '
                     f'fill="{muted}" font-family="sans-serif" font-size="12">{i}</text>')
    parts.append(f'<text x="{(left + right) / 2}" y="{bottom2 + 40}" text-anchor="middle" '
                 f'fill="{fg}" font-family="sans-serif" font-size="13">Decision within episode (500 MD steps each)</text>')
    parts.append(f'<text x="{left}" y="{top2 - 10}" fill="{fg}" '
                 'font-family="sans-serif" font-size="13">Rolling 10-decision reward (s/segment)</text>')
    parts.append(f'<text x="{right}" y="{top2 + 18}" text-anchor="end" fill="{exploratory}" '
                 'font-family="sans-serif" font-size="13">Episodes 8–11</text>')
    parts.append(f'<text x="{right}" y="{top2 + 36}" text-anchor="end" fill="{learned}" '
                 'font-family="sans-serif" font-size="13">Episodes 32–35</text>')
    note = 'Descriptive training trace only — not a paired fixed-policy evaluation.'
    parts.append(f'<text x="{left}" y="{height - 20}" fill="{muted}" '
                 f'font-family="sans-serif" font-size="13">{escape(note)}</text>')
    parts.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(parts) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
