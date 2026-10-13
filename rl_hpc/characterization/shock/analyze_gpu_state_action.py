#!/usr/bin/env python3
"""Analyze GPU Shock state/action forks by imbalance and performance state."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

FACTORS = (None, .5, .75, 1., 1.25, 1.5)
REGIMES = ("idle", "cpu_contention", "gpu_contention")

def label(x): return "skip" if x is None else f"time {x:g}"
def mean(xs): return statistics.mean(xs)
def std(xs): return statistics.stdev(xs)
def imb(state, key):
    item=state["previous_segment"][key]
    return item["max"]/item["mean"]
def perf(state, key):
    seg=state["previous_segment"]
    return seg["timing_avg_seconds"][key]/seg["steps"]

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    rows=[json.loads(x) for x in (a.input/"measurements.jsonl").read_text().splitlines()]
    assert len(rows)==720 and all(r["safe"] for r in rows)
    cells=[]
    for phase in range(1,9):
      for regime in REGIMES:
       for factor in FACTORS:
        group=sorted((r for r in rows if r["phase"]==phase and r["regime"]==regime and r["action"]["factor"]==factor),key=lambda r:r["repetition"])
        assert len(group)==5
        runt=[r["runtime_seconds"] for r in group]
        cells.append(dict(phase=phase,regime=regime,factor=factor,n=5,mean=mean(runt),std=std(runt),cv=std(runt)/mean(runt)*100,
          source_atom=mean([imb(r["source_state"],"nlocal") for r in group]),source_neighbor=mean([imb(r["source_state"],"neighs") for r in group]),source_ghost=mean([imb(r["source_state"],"nghost") for r in group]),
          next_atom=mean([imb(r["next_state"],"nlocal") for r in group]),next_neighbor=mean([imb(r["next_state"],"neighs") for r in group]),next_ghost=mean([imb(r["next_state"],"nghost") for r in group]),
          source_pair=mean([perf(r["source_state"],"pair") for r in group]),source_comm=mean([perf(r["source_state"],"comm") for r in group]),
          next_pair=mean([perf(r["next_state"],"pair") for r in group]),next_comm=mean([perf(r["next_state"],"comm") for r in group]),samples=runt))
    summary={"cells":cells,"regimes":{}}
    for regime in REGIMES:
      rc=[c for c in cells if c["regime"]==regime]
      best_by_state={str(p):min((c for c in rc if c["phase"]==p),key=lambda c:c["mean"])["factor"] for p in range(1,9)}
      totals={str(f):sum(c["mean"] for c in rc if c["factor"]==f) for f in FACTORS}
      fixed_key=min(totals,key=totals.get);fixed=None if fixed_key=="None" else float(fixed_key)
      oracle=sum(min(c["mean"] for c in rc if c["phase"]==p) for p in range(1,9))
      summary["regimes"][regime]=dict(best_by_state=best_by_state,best_fixed=fixed,fixed_total=totals[fixed_key],oracle_total=oracle,oracle_gain_percent=(totals[fixed_key]-oracle)/totals[fixed_key]*100)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# GPU Shock state × action characterization","","MPI32 / 4 GPUs / CUDA MPS、各cell 5反復。actionはskipまたは`balance ... weight time factor`。","","## Runtime matrix","","|State|Regime|Best action|Mean s|Std s|CV %|Runner-up gap %|","|---:|---|---|---:|---:|---:|---:|"]
    for phase in range(1,9):
      for regime in REGIMES:
        cs=sorted((c for c in cells if c["phase"]==phase and c["regime"]==regime),key=lambda c:c["mean"]);b=cs[0]
        lines.append(f"|S{phase}|{regime}|{label(b['factor'])}|{b['mean']:.6f}|{b['std']:.6f}|{b['cv']:.2f}|{(cs[1]['mean']-b['mean'])/b['mean']*100:.2f}|")
    lines += ["","## Best Fixedと記述的oracle","","|Regime|Best Fixed|Fixed total s|Oracle total s|Oracle gain %|","|---|---|---:|---:|---:|"]
    for regime,d in summary["regimes"].items(): lines.append(f"|{regime}|{label(d['best_fixed'])}|{d['fixed_total']:.6f}|{d['oracle_total']:.6f}|{d['oracle_gain_percent']:.3f}|")
    lines += ["","oracleは同じ測定値上で選択・評価した記述値であり、未知状態への汎化性能ではない。"]
    (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary["regimes"],indent=2))
if __name__=="__main__":main()
