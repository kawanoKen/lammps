#!/usr/bin/env python3
"""Create CPU-report-style figures for the GPU Shock state/action matrix."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

FACTORS=(None,.5,.75,1.,1.25,1.5);REGIMES=('idle','cpu_contention','gpu_contention')
LABELS=('skip','0.50','0.75','1.00','1.25','1.50');COLORS=dict(idle='tab:blue',cpu_contention='tab:orange',gpu_contention='tab:red')
def imb(s,k):q=s['previous_segment'][k];return q['max']/q['mean']
def perf(s,k):q=s['previous_segment'];return q['timing_avg_seconds'][k]/q['steps']
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(x) for x in (a.input/'measurements.jsonl').read_text().splitlines()]
 fig,axes=plt.subplots(2,4,figsize=(17,8),sharey=False)
 for phase,ax in enumerate(axes.flat,1):
  for reg in REGIMES:
   means=[];stds=[]
   for f in FACTORS:
    v=[r['runtime_seconds'] for r in rows if r['phase']==phase and r['regime']==reg and r['action']['factor']==f]
    means.append(np.mean(v));stds.append(np.std(v,ddof=1))
   ax.errorbar(range(6),means,yerr=stds,marker='o',capsize=3,label=reg,color=COLORS[reg])
  ax.set(title=f'S{phase}',xticks=range(6),xticklabels=LABELS,xlabel='skip / time factor',ylabel='action + 500 steps (s)');ax.grid(alpha=.25)
 axes.flat[0].legend(fontsize=8);fig.suptitle('GPU Shock: runtime by state, action, and resource regime');fig.tight_layout();fig.savefig(a.output/'phase_action_runtimes.png',dpi=180);plt.close(fig)
 specs=[('imbalance',['atom','neighbor','ghost']),('performance',['pair','neigh','comm','runtime'])]
 metadata={}
 for kind,features in specs:
  vectors=[];items=[]
  for r in rows:
   sv=([imb(r['source_state'],k) for k in ('nlocal','neighs','nghost')] if kind=='imbalance' else [perf(r['source_state'],k) for k in ('pair','neigh','comm')]+[r['source_probe_seconds']])
   nv=([imb(r['next_state'],k) for k in ('nlocal','neighs','nghost')] if kind=='imbalance' else [perf(r['next_state'],k) for k in ('pair','neigh','comm')]+[r['runtime_seconds']])
   vectors += [sv,nv];items.append((r,sv,nv))
  z=np.asarray(vectors);mu=z.mean(0);scale=z.std(0);scale[scale==0]=1;u,s,v=np.linalg.svd((z-mu)/scale,full_matrices=False);ratio=s*s/(s*s).sum()
  fig,axes=plt.subplots(2,4,figsize=(16,8))
  for phase,ax in enumerate(axes.flat,1):
   for reg in REGIMES:
    group=[x for x in items if x[0]['phase']==phase and x[0]['regime']==reg]
    src=np.mean([x[1] for x in group],0);sc=((src-mu)/scale)@v[:2].T
    ax.scatter(*sc,marker='D',s=55,color=COLORS[reg])
    for f in FACTORS:
     g=[x for x in group if x[0]['action']['factor']==f];dst=np.mean([x[2] for x in g],0);dc=((dst-mu)/scale)@v[:2].T
     ax.plot([sc[0],dc[0]],[sc[1],dc[1]],color=COLORS[reg],alpha=.35,lw=1);ax.scatter(*dc,s=28,color=COLORS[reg],alpha=.75)
   ax.set(title=f'S{phase}',xlabel=f'PC1 {ratio[0]:.1%}',ylabel=f'PC2 {ratio[1]:.1%}');ax.grid(alpha=.2)
  fig.suptitle(f'{kind.capitalize()} state transitions: diamond=source, circles=action-conditioned next states');fig.tight_layout();fig.savefig(a.output/f'state_transition_{kind}.png',dpi=180);plt.close(fig)
  metadata[kind]=dict(features=features,mean=mu.tolist(),scale=scale.tolist(),components=v[:2].tolist(),variance_ratio=ratio[:2].tolist())
 (a.output/'state_transition_pca_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':main()
