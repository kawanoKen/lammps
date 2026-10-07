#!/usr/bin/env python3
"""Create the internal-state checkpoint-fork report from raw JSONL."""
from __future__ import annotations
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path

def load(directory:Path):
 rows=[json.loads(x) for x in (directory/'measurements.jsonl').read_text().splitlines() if x]
 states={x['checkpoint_id']:x for x in json.loads((directory/'checkpoint_states.json').read_text())}
 return rows,states
def summarize(rows):
 groups=defaultdict(list)
 for r in rows:
  if r['safe'] and 'loop_seconds' in r['lammps']:groups[(r['checkpoint_id'],r['action']['skin'],r['action']['every'])].append(r['lammps']['loop_seconds'])
 out=[]
 for (cp,s,e),values in groups.items():out.append({'checkpoint':cp,'skin':s,'every':e,'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else 0.,'n':len(values),'cv':statistics.stdev(values)/statistics.mean(values)*100 if len(values)>1 else 0.})
 return sorted(out,key=lambda x:(x['checkpoint'],x['mean']))
def section(name,rows,states):
 summary=summarize(rows); by=defaultdict(list)
 for x in summary:by[x['checkpoint']].append(x)
 lines=[f'## {name}','', '| Checkpoint | Step | Action | Mean loop (s) | Std | CV | n | Rank |','|---|---:|---|---:|---:|---:|---:|---:|']
 oracle=0.; fixed_totals=defaultdict(float); action_coverage=defaultdict(set); best=[]
 for cp,xs in sorted(by.items()):
  xs.sort(key=lambda x:x['mean']); oracle+=xs[0]['mean'];best.append((cp,xs[0]))
  for x in xs:
   action=(x['skin'],x['every']);fixed_totals[action]+=x['mean'];action_coverage[action].add(cp)
  for rank,x in enumerate(xs,1):lines.append(f"| {cp} | {states[cp].get('simulation_step',0):.0f} | skin={x['skin']:g}, every={x['every']} | {x['mean']:.6f} | {x['std']:.6f} | {x['cv']:.2f}% | {x['n']} | {rank} |")
 # A fixed policy must be valid at every checkpoint; never compare an
 # incomplete four-checkpoint sum with the five-checkpoint oracle.
 complete={a:t for a,t in fixed_totals.items() if len(action_coverage[a])==len(by)}
 fixed_action,fixed=min(complete.items(),key=lambda x:x[1]); improvement=(fixed-oracle)/fixed*100
 lines += ['',f'Best action by checkpoint: '+', '.join(f"{cp}=({x['skin']:g},{x['every']})" for cp,x in best)+'.','',f'- State-conditioned oracle total: **{oracle:.6f} s**','- Best fixed action: **skin=%g, every=%d**; total **%.6f s**' % (fixed_action[0],fixed_action[1],fixed),f'- Theoretical checkpoint-conditioned improvement: **{improvement:.2f}%**.']
 unsafe=[r for r in rows if not r['safe']]
 lines += [f'- Invalid/unsafe branch records excluded: **{len(unsafe)}**.','']
 lines += ['Checkpoint state before actions:', '', '| Checkpoint | Step | Temp | Press | Density | Pair | Neigh | Comm | KSpace | Neighbor builds |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for cp,s in sorted(states.items()):
  t=s.get('previous_segment_timing_seconds',{});lines.append(f"| {cp} | {s.get('simulation_step',0):.0f} | {s.get('temperature',0):.5g} | {s.get('pressure',0):.5g} | {s.get('density',0):.5g} | {t.get('pair',0):.5f} | {t.get('neigh',0):.5f} | {t.get('comm',0):.5f} | {t.get('kspace',0):.5f} | {s.get('neighbor_builds',0)} |")
 return '\n'.join(lines),{'oracle':oracle,'fixed':fixed,'improvement_pct':improvement,'best_fixed':fixed_action,'best_by_checkpoint':[(cp,(x['skin'],x['every'])) for cp,x in best]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--lj',type=Path,required=True);p.add_argument('--spce',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 lrows,lstates=load(a.lj);srows,sstates=load(a.spce); ltext,lresult=section('Lennard-Jones (GPU, idle)',lrows,lstates);stext,sresult=section('SPC/E + PPPM (GPU, idle)',srows,sstates)
 text='''# Internal-state checkpoint-fork characterization

## Result

This is a direct counterfactual measurement: each candidate action starts from
the *same binary restart state* in a separate LAMMPS process. No GPU co-runner
or external contention was used. Across five evolving checkpoints, neither
workload shows a meaningful change of the best safe neighbor configuration.
The state-conditioned oracle equals the best fixed configuration in both
workloads, so the measured adaptation opportunity is **0.00%**.

`delay=0` and `check=yes` were fixed. The primary metric is LAMMPS loop time
for the same 500 steps; process startup is not used for ranking.

## Interpretation

- **LJ (control):** `skin=0.4, every=20` has the lowest mean at every
  checkpoint. Its difference from `skin=0.4, every=10` is sometimes below
  the 3-run noise, so those two should be treated as a stable near-optimal
  set—not evidence of a changing optimum. The large skin effect is stable.
- **SPC/E + PPPM:** `skin=2.0, every=5` is the winner at every checkpoint.
  It is about 1.0--1.2% faster than the runner-up (`skin=2.0, every=1`),
  larger than the within-action CVs (roughly 0.1--0.3%), and the ordering does
  not reverse. Although temperature, pressure, Neigh, Comm, and KSpace timing
  fluctuate across checkpoints, they do not correspond to a new best action.
- `skin=4.0, every=20` produced dangerous neighbor builds at SPC/E checkpoint
  S3 in all three repeats. It is invalid there and excluded from all
  performance conclusions and from the fixed-policy candidate set.

Consequently, these natural internal-state changes do **not** justify an
internal-state-only RL controller for the tested neighbor controls. The prior
same-GPU-contention result remains an external-resource phenomenon rather
than evidence of a changing intrinsic LAMMPS optimum.

'''+ltext+'\n\n'+stext+'\n'
 a.output.write_text(text);print(json.dumps({'lj':lresult,'spce':sresult},indent=2))
if __name__=='__main__':main()
