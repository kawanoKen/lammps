#!/usr/bin/env python3
"""Summarize paired live evaluation without using hidden labels as features."""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np

def interval(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    if len(x)<2:return [float(x.mean()),float(x.mean())]
    means=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(5000)])
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
 rows=[json.loads(x) for x in (a.directory/'live_transitions.jsonl').read_text().splitlines() if x]
 by=collections.defaultdict(lambda:collections.defaultdict(float))
 unsafe=collections.Counter()
 for r in rows:
  # policy_worker names each output `episode-NNN-method`; recover the paired
  # schedule identity rather than treating each method as a separate episode.
  paired_episode=r['episode_id'].rsplit('-',1)[0]
  by[r['evaluation_method']][paired_episode]+=r['segment_runtime_seconds'];unsafe[r['evaluation_method']]+=r['unsafe']
 methods=sorted(by); common=sorted(set.intersection(*(set(by[m]) for m in methods))); rng=np.random.default_rng(20260929)
 fixed=np.array([by['fixed'][e] for e in common]);oracle=np.array([by['oracle'][e] for e in common]);opportunity=fixed-oracle
 report={'episodes':common,'methods':{}}
 for m in methods:
  values=np.array([by[m][e] for e in common]);diff=values-fixed
  recovery=(fixed-values).sum()/opportunity.sum() if opportunity.sum()>0 else None
  regret=values-oracle
  report['methods'][m]={'mean_seconds':float(values.mean()),'std_seconds':float(values.std(ddof=1)) if len(values)>1 else 0.0,'mean_ci95':interval(values,rng),'vs_fixed_mean_seconds':float(diff.mean()),'vs_fixed_ci95':interval(diff,rng),'oracle_regret_mean_seconds':float(regret.mean()),'recovery':None if recovery is None else float(recovery),'unsafe':int(unsafe[m])}
 (a.directory/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
