#!/usr/bin/env python3
"""Render GPU Shock figures with Pillow when matplotlib is unavailable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FACTORS = (None, .5, .75, 1., 1.25, 1.5)
LABELS = ("skip", "0.50", "0.75", "1.00", "1.25", "1.50")
REGIMES = ("idle", "cpu_contention", "gpu_contention")
COLORS = {"idle": (31, 119, 180), "cpu_contention": (255, 127, 14),
          "gpu_contention": (214, 39, 40)}


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def text_center(draw, xy, value, fnt, fill=(25, 25, 25)):
    box = draw.textbbox((0, 0), value, font=fnt)
    draw.text((xy[0]-(box[2]-box[0])/2, xy[1]-(box[3]-box[1])/2),
              value, font=fnt, fill=fill)


def panel(draw, box, title, xlabel, ylabel, xlim, ylim):
    x0, y0, x1, y1 = box
    left, right, top, bottom = 105, 35, 55, 90
    plot = (x0+left, y0+top, x1-right, y1-bottom)
    px0, py0, px1, py1 = plot
    draw.rectangle(plot, outline=(70, 70, 70), width=2)
    for frac in (.25, .5, .75):
        y = py1-frac*(py1-py0)
        draw.line((px0, y, px1, y), fill=(220, 220, 220), width=1)
    text_center(draw, ((x0+x1)/2, y0+24), title, font(25, True))
    text_center(draw, ((px0+px1)/2, y1-27), xlabel, font(18))
    draw.text((x0+6, y0+8), ylabel, font=font(17), fill=(35, 35, 35))
    for frac in (0, .25, .5, .75, 1):
        value = ylim[0]+frac*(ylim[1]-ylim[0])
        y = py1-frac*(py1-py0)
        draw.line((px0-7, y, px0, y), fill=(50, 50, 50), width=2)
        label = f"{value:.2f}"
        box2 = draw.textbbox((0, 0), label, font=font(15))
        draw.text((px0-12-(box2[2]-box2[0]), y-9), label, font=font(15), fill=(40,40,40))
    return plot


def scale(value, limits, pixels, invert=False):
    fraction = (value-limits[0])/(limits[1]-limits[0])
    if invert:
        fraction = 1-fraction
    return pixels[0]+fraction*(pixels[1]-pixels[0])


def imbalance(state, key):
    item = state["previous_segment"][key]
    return item["max"] / item["mean"]


def performance(state, key):
    segment = state["previous_segment"]
    return segment["timing_avg_seconds"][key] / segment["steps"]


def runtime_figure(rows, output):
    width, height = 3060, 1440
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    text_center(draw, (width/2, 34),
                "GPU Shock: runtime by state, action, and resource regime", font(30, True))
    margin, gap, top = 45, 35, 75
    pw = (width-2*margin-3*gap)/4
    ph = (height-top-45-gap)/2
    for phase in range(1, 9):
        row, col = divmod(phase-1, 4)
        box = (margin+col*(pw+gap), top+row*(ph+gap),
               margin+col*(pw+gap)+pw, top+row*(ph+gap)+ph)
        subset = [r for r in rows if r["phase"] == phase]
        values = [r["runtime_seconds"] for r in subset]
        lo, hi = min(values), max(values)
        pad = max((hi-lo)*.12, .04)
        plot = panel(draw, box, f"S{phase}", "skip / time factor",
                     "action + 500 steps (s)", (-.3, 5.3), (lo-pad, hi+pad))
        px0, py0, px1, py1 = plot
        for regime in REGIMES:
            points = []
            for index, factor in enumerate(FACTORS):
                sample = [r["runtime_seconds"] for r in subset
                          if r["regime"] == regime and r["action"]["factor"] == factor]
                mean, sd = float(np.mean(sample)), float(np.std(sample, ddof=1))
                x = scale(index, (-.3, 5.3), (px0, px1))
                y = scale(mean, (lo-pad, hi+pad), (py0, py1), True)
                yl = scale(mean-sd, (lo-pad, hi+pad), (py0, py1), True)
                yh = scale(mean+sd, (lo-pad, hi+pad), (py0, py1), True)
                draw.line((x, yl, x, yh), fill=COLORS[regime], width=2)
                draw.line((x-5, yl, x+5, yl), fill=COLORS[regime], width=2)
                draw.line((x-5, yh, x+5, yh), fill=COLORS[regime], width=2)
                points.append((x, y))
            draw.line(points, fill=COLORS[regime], width=3)
            for x, y in points:
                draw.ellipse((x-5, y-5, x+5, y+5), fill=COLORS[regime])
        for index, label in enumerate(LABELS):
            x = scale(index, (-.3, 5.3), (px0, px1))
            text_center(draw, (x, py1+22), label, font(14))
    x = 85
    for regime in REGIMES:
        draw.line((x, 37, x+35, 37), fill=COLORS[regime], width=4)
        draw.ellipse((x+13, 32, x+23, 42), fill=COLORS[regime])
        draw.text((x+43, 25), regime, font=font(17), fill=(30,30,30))
        x += 220
    image.save(output)


def transition_figure(rows, output, kind):
    items, vectors = [], []
    for record in rows:
        if kind == "imbalance":
            source = [imbalance(record["source_state"], key)
                      for key in ("nlocal", "neighs", "nghost")]
            target = [imbalance(record["next_state"], key)
                      for key in ("nlocal", "neighs", "nghost")]
        else:
            source = [performance(record["source_state"], key)
                      for key in ("pair", "neigh", "comm")] + [record["source_probe_seconds"]]
            target = [performance(record["next_state"], key)
                      for key in ("pair", "neigh", "comm")] + [record["runtime_seconds"]]
        vectors.extend((source, target))
        items.append((record, np.asarray(source), np.asarray(target)))
    values = np.asarray(vectors)
    mean, std = values.mean(0), values.std(0)
    std[std == 0] = 1
    _, singular, components = np.linalg.svd((values-mean)/std, full_matrices=False)
    ratio = singular**2/(singular**2).sum()
    projected = ((values-mean)/std) @ components[:2].T
    xlim = (float(projected[:,0].min()), float(projected[:,0].max()))
    ylim = (float(projected[:,1].min()), float(projected[:,1].max()))
    xpad, ypad = max((xlim[1]-xlim[0])*.08, .1), max((ylim[1]-ylim[0])*.08, .1)
    xlim = (xlim[0]-xpad, xlim[1]+xpad); ylim = (ylim[0]-ypad, ylim[1]+ypad)
    width, height = 2880, 1440
    image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
    title = ("Imbalance" if kind == "imbalance" else "Performance") + \
            " state transitions: diamond=source, circles=action-conditioned next states"
    text_center(draw, (width/2, 34), title, font(29, True))
    margin, gap, top = 40, 32, 75
    pw=(width-2*margin-3*gap)/4; ph=(height-top-45-gap)/2
    for phase in range(1, 9):
        rr, cc = divmod(phase-1, 4)
        box=(margin+cc*(pw+gap), top+rr*(ph+gap), margin+cc*(pw+gap)+pw, top+rr*(ph+gap)+ph)
        plot=panel(draw, box, f"S{phase}", f"PC1 {ratio[0]:.1%}",
                   f"PC2 {ratio[1]:.1%}", xlim, ylim)
        px0,py0,px1,py1=plot
        for regime in REGIMES:
            group=[x for x in items if x[0]["phase"]==phase and x[0]["regime"]==regime]
            src=np.mean([x[1] for x in group],axis=0); sp=((src-mean)/std)@components[:2].T
            sx=scale(sp[0],xlim,(px0,px1)); sy=scale(sp[1],ylim,(py0,py1),True)
            for factor in FACTORS:
                matches=[x for x in group if x[0]["action"]["factor"]==factor]
                dst=np.mean([x[2] for x in matches],axis=0); dp=((dst-mean)/std)@components[:2].T
                dx=scale(dp[0],xlim,(px0,px1)); dy=scale(dp[1],ylim,(py0,py1),True)
                draw.line((sx,sy,dx,dy),fill=tuple(list(COLORS[regime])+[80])[:3],width=2)
                draw.ellipse((dx-5,dy-5,dx+5,dy+5),fill=COLORS[regime])
            draw.polygon([(sx,sy-8),(sx+8,sy),(sx,sy+8),(sx-8,sy)],fill=COLORS[regime])
    image.save(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (args.input/"measurements.jsonl").read_text().splitlines()]
    runtime_figure(rows,args.output/"phase_action_runtimes.png")
    transition_figure(rows,args.output/"state_transition_imbalance.png","imbalance")
    transition_figure(rows,args.output/"state_transition_performance.png","performance")


if __name__ == "__main__":
    main()
