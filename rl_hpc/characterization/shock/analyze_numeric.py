#!/usr/bin/env python3
"""Numerical response surfaces and independently confirmed rank reversals.

Selection uses discovery only. Confirmation jobs are frozen before collection.
Bootstrap intervals quantify repetition variability, not cross-trajectory
generalization. Single-trajectory conclusions remain conditional.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import statistics as st
import time

import numeric_sweep as sweep


def group(rows, phase):
    result={}
    for r in rows:
        if r['phase']==phase and r['safe']:
            result.setdefault((r['checkpoint'],r['skin'],r['factor']),[]).append(r['times']['ACTION_WALL_SECONDS'])
    return result


def difference_ci(a,b):
    rng=random.Random(8231)
    samples=sorted(st.mean(rng.choices(a,k=len(a)))-st.mean(rng.choices(b,k=len(b))) for _ in range(10000))
    return [samples[250],samples[9749]]


def analyze(out):
    rows=[json.loads(l) for l in (out/'measurements.jsonl').read_text().splitlines()]
    g=group(rows,'main')
    if len(g)!=180 or any(len(v)!=5 for v in g.values()):
        raise RuntimeError('Expected all 180 cells with five repetitions')
    actions=list(itertools.product(sweep.SKINS,sweep.FACTORS))
    states=sorted({k[0] for k in g})
    means={k:st.mean(v) for k,v in g.items()}
    winners={s:min(actions,key=lambda a:means[(s,*a)]) for s in states}
    fixed=min(actions,key=lambda a:sum(means[(s,*a)] for s in states))
    fixed_sum=sum(means[(s,*fixed)] for s in states)
    oracle_sum=sum(means[(s,*winners[s])] for s in states)
    with (out/'matrix.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['state','skin','factor','n','mean_s','std_s','cv_percent','rank','regret_percent'])
        for s in states:
            order=sorted(actions,key=lambda a:means[(s,*a)])
            for rank,a in enumerate(order,1):
                v=g[(s,*a)];m=st.mean(v)
                writer.writerow([s,*a,len(v),m,st.stdev(v),100*st.stdev(v)/m,rank,100*(m/means[(s,*order[0])]-1)])
    # Select one strongest opposite ordering, and freeze it for new trials.
    candidates=[]
    for s,t in itertools.combinations(states,2):
        a,b=winners[s],winners[t]
        if a==b: continue
        gap_s=means[(s,*b)]/means[(s,*a)]-1
        gap_t=means[(t,*a)]/means[(t,*b)]-1
        if min(gap_s,gap_t)>0:
            candidates.append((min(gap_s,gap_t),s,t,a,b))
    plan=[]
    if candidates:
        _,s,t,a,b=max(candidates)
        plan=[['confirmation',state,*action,rep] for rep in range(1,11) for state in (s,t) for action in (a,b)]
        random.Random(995).shuffle(plan)
    plan_path=out/'confirmation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise RuntimeError('Discovery changed after confirmation selection')
    plan_path.write_text(json.dumps(plan,indent=2))
    report=['# Shock: numerical action sweep','',
            '32 MPI ranks; six reused physical checkpoints; common 50-step warm-up; 500 measured steps.',
            'Actions: skin × neighbor-weight factor. CVCF/history diagnostics excluded in every fork.',
            'Total includes list setup, one-time balance, and simulation. No time-weight action is used.','',
            '| State | Best skin | Best factor | Mean seconds | Std | CV % |',
            '|---|---:|---:|---:|---:|---:|']
    for s in states:
        a=winners[s];v=g[(s,*a)];m=st.mean(v)
        report.append(f'| {s} | {a[0]} | {a[1]} | {m:.6f} | {st.stdev(v):.6f} | {100*st.stdev(v)/m:.3f} |')
    report += ['',f'Best fixed discovery action: skin={fixed[0]}, factor={fixed[1]}.',
               f'Summed fixed means: {fixed_sum:.6f} s; selected oracle: {oracle_sum:.6f} s.',
               f'Descriptive improvement (fixed-oracle)/fixed: {100*(fixed_sum-oracle_sum)/fixed_sum:.3f}%.',
               'This is a selection-biased discovery estimate, not a live-policy gain or a proven upper bound.',
               'Full state-action means, variability, ranks and regrets: matrix.csv.',
               'Response surfaces: response.svg (per-state relative runtime; numbers are seconds).','']
    confirmed=group(rows,'confirmation')
    if plan:
        _,s,t,a,b=max(candidates)
        report += [f'Frozen confirmation pair: states {s}/{t}, actions {a}/{b}.',
                   'Positive A-minus-B means B is faster; negative means A is faster.']
        intervals=[]
        for state in (s,t):
            va,vb=confirmed.get((state,*a),[]),confirmed.get((state,*b),[])
            if len(va)==10 and len(vb)==10:
                ci=difference_ci(va,vb);intervals.append(ci)
                report.append(f'{state}: A-B={st.mean(va)-st.mean(vb):.6f} s; bootstrap 95% CI [{ci[0]:.6f}, {ci[1]:.6f}].')
            else: report.append(f'{state}: confirmation pending ({len(va)}/10 A, {len(vb)}/10 B).')
        if len(intervals)==2:
            reversal=(intervals[0][1]<0 and intervals[1][0]>0) or (intervals[1][1]<0 and intervals[0][0]>0)
            report.append(f'Independent confirmation of opposite ordering with both intervals excluding zero: {reversal}.')
    else: report.append('All states select the same discovery winner; no winner-reversal candidate for confirmation.')
    report += ['', 'Interpretation is limited to these checkpoints and the sampled parameter grid.',
               'Boundary optima require range extension before claiming an optimum.',
               'No RL or SMDP training was performed.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    # Dependency-free SVG heatmaps, with absolute numbers and relative colors.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="660">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="20" y="25" font-size="18">500-step total seconds; columns: skin; rows: neighbor weight</text>']
    for idx,s in enumerate(states):
        ox=20+(idx%3)*365;oy=65+(idx//3)*295
        best=means[(s,*winners[s])]
        svg.append(f'<text x="{ox}" y="{oy}" font-size="18">{s}</text>')
        for j,skin in enumerate(sweep.SKINS):
            svg.append(f'<text x="{ox+58+j*58}" y="{oy+22}" font-size="12">{skin}</text>')
        for i,factor in enumerate(sweep.FACTORS):
            y=oy+30+i*35
            svg.append(f'<text x="{ox}" y="{y+22}" font-size="12">{factor}</text>')
            for j,skin in enumerate(sweep.SKINS):
                value=means[(s,skin,factor)];v=min(1,(value/best-1)/.4)
                color=f'rgb({int(210+40*v)},{int(240-130*v)},{int(240-130*v)})'
                x=ox+45+j*58
                svg.append(f'<rect x="{x}" y="{y}" width="57" height="34" fill="{color}"/>')
                svg.append(f'<text x="{x+3}" y="{y+22}" font-size="12">{value:.3f}</text>')
    svg.append('</svg>');(out/'response.svg').write_text('\n'.join(svg))
    return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();analyze(a.output.resolve())


if __name__=='__main__':main()
