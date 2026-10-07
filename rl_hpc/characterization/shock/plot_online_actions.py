#!/usr/bin/env python3
"""Plot selected skin and balance-weight actions through online training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SKIN_COLORS = {
    '0.25': '#d9ecf4', '0.35': '#a8d3e4', '0.5': '#2b85ae',
    '0.7': '#19506f', '1.0': '#082d42',
}
FACTOR_COLORS = {
    '0.5': '#642600', '0.65': '#a04713', '0.8': '#d57d35',
    '1.0': '#e9c29a', '1.2': '#5a8d39', '1.5': '#245c34',
}
EPSILON = '#d5d8dc'
SKIP = '#1b1b1b'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    with (root / 'reward_trace.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    total, first, decisions = (manifest['episodes'], manifest['explore_episodes'],
                               manifest['decisions'])
    learning = [r for r in rows if int(r['episode']) >= first]
    if len(learning) != (total - first) * decisions:
        raise SystemExit('expected a fully audited learning trace')

    width, height = 1200, 770
    left, right = 95, 1045
    panels = [(102, 347, 'skin', SKIN_COLORS),
              (418, 663, 'neighbor_factor', FACTOR_COLORS)]
    n_episodes = total - first
    cell_w = (right - left) / decisions
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Online shock policy action choices</title>',
        '<desc id="desc">Two aligned heatmaps show skin and neighbor weight selected '
        'at each of 120 decisions in episodes 12 through 35. Gray cells are '
        'epsilon-random exploration. Later episodes show skin 0.5 early, with '
        'some skin 0.7 late, and weight factors moving from 1.2 or 1.5 early '
        'to 0.5 late.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="95" y="32" fill="#192637" font-family="sans-serif" '
        'font-size="23" font-weight="600">Online action choices</text>',
        '<text x="95" y="56" fill="#546579" font-family="sans-serif" '
        'font-size="13">Each cell = one 500-step decision; gray = epsilon-random, not the greedy policy</text>',
    ]
    for top, bottom, field, colors in panels:
        cell_h = (bottom - top) / n_episodes
        label = 'Neighbor skin' if field == 'skin' else 'Balance weight neigh factor'
        parts.append(f'<text x="{left}" y="{top - 10}" fill="#192637" '
                     f'font-family="sans-serif" font-size="15">{label}</text>')
        for row in learning:
            ep = int(row['episode'])
            decision = int(row['decision'])
            if row['action_source'] != 'greedy':
                color = EPSILON
            elif field == 'neighbor_factor' and row['balance'] == 'False':
                color = SKIP
            else:
                value = row[field]
                if value not in colors:
                    raise SystemExit(f'unexpected {field} value: {value}')
                color = colors[value]
            x = left + decision * cell_w
            y = top + (ep - first) * cell_h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + .12:.2f}" '
                         f'height="{cell_h + .12:.2f}" fill="{color}"/>')
        for decision in range(0, decisions + 1, 20):
            x = left + decision * cell_w
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                         f'y2="{bottom}" stroke="#ffffff" stroke-width="1.2"/>')
            if decision < decisions:
                parts.append(f'<text x="{x:.2f}" y="{bottom + 20}" '
                             f'fill="#546579" font-family="sans-serif" font-size="12" '
                             f'text-anchor="middle">{decision}</text>')
        parts.append(f'<text x="{right}" y="{bottom + 20}" '
                     f'fill="#546579" font-family="sans-serif" font-size="12" '
                     f'text-anchor="middle">{decisions}</text>')
        for ep in range(first, total, 4):
            y = top + (ep - first + .5) * cell_h
            parts.append(f'<text x="{left - 11}" y="{y + 4:.2f}" '
                         f'fill="#546579" font-family="sans-serif" font-size="12" '
                         f'text-anchor="end">{ep}</text>')
        parts.append(f'<text x="{left - 11}" y="{bottom - cell_h / 2 + 4:.2f}" '
                     f'fill="#546579" font-family="sans-serif" font-size="12" '
                     f'text-anchor="end">{total - 1}</text>')
        parts.append(f'<text x="{(left + right) / 2}" y="{bottom + 40}" '
                     f'fill="#192637" font-family="sans-serif" font-size="13" '
                     'text-anchor="middle">Decision within episode</text>')
        legend_y = top + 2
        for value, color in colors.items():
            parts.append(f'<rect x="1065" y="{legend_y}" width="17" height="14" '
                         f'fill="{color}"/>')
            parts.append(f'<text x="1089" y="{legend_y + 12}" '
                         f'fill="#192637" font-family="sans-serif" font-size="13">{value}</text>')
            legend_y += 24
        if field == 'neighbor_factor':
            parts.append(f'<rect x="1065" y="{legend_y}" width="17" height="14" '
                         f'fill="{SKIP}"/>')
            parts.append(f'<text x="1089" y="{legend_y + 12}" '
                         'fill="#192637" font-family="sans-serif" font-size="13">skip</text>')
    parts.append(f'<rect x="1065" y="{height - 48}" width="17" height="14" '
                 f'fill="{EPSILON}"/>')
    parts.append(f'<text x="1089" y="{height - 36}" fill="#192637" '
                 'font-family="sans-serif" font-size="13">epsilon</text>')
    parts.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(parts) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
