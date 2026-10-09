#!/usr/bin/env python3
"""Describe matched checkpoint/action idle-contention observations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FACTORS = (None, .5, .75, 1., 1.25, 1.5)
REGIMES = ('idle', 'contention')
METRICS = ('runtime_seconds', 'reward', 'atom_imbalance', 'neighbor_imbalance',
           'ghost_imbalance', 'pair_per_step', 'neigh_per_step', 'comm_per_step')


def values(row):
    m = row['next_state']['lammps']
    return dict(runtime_seconds=row['runtime_seconds'], reward=row['reward'],
                atom_imbalance=m['nlocal']['max']/m['nlocal']['mean'],
                neighbor_imbalance=m['neighs']['max']/m['neighs']['mean'],
                ghost_imbalance=m['nghost']['max']/m['nghost']['mean'],
                **{f'{name}_per_step': m['timing_avg_seconds'][name]/500
                   for name in ('pair', 'neigh', 'comm')})


def stats(a):
    a=np.asarray(a, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std(ddof=1)),
                cv_percent=float(a.std(ddof=1)/abs(a.mean())*100) if a.mean() else None,
                minimum=float(a.min()), maximum=float(a.max()), samples=a.tolist())


def label(f):
    return 'skip' if f is None else f'{f:.2f}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, choices=(4,5), default=5)
    args=p.parse_args()
    rows=[json.loads(s) for s in (args.input/'measurements.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['repetition'] < args.repetitions]
    count=96*args.repetitions
    assert len(rows)==count and all(r['safe'] and r['returncode']==0 for r in rows)
    source_manifest=json.loads((args.input/'manifest.json').read_text())
    assert [r['key'] for r in rows] == source_manifest['schedule'][:count]
    cells=[]
    for phase in range(1,9):
        group=[r for r in rows if r['phase']==phase]
        assert len({tuple(r['pre_thermo_reconstructed']) for r in group})==1
        assert len({r['reconstructed_mesh_sha256'] for r in group})==1
        for factor in FACTORS:
            grouped={regime:sorted([r for r in group if r['action']['factor']==factor and r['regime']==regime],
                                   key=lambda r:r['repetition']) for regime in REGIMES}
            assert all([r['repetition'] for r in grouped[g]]==list(range(args.repetitions)) for g in REGIMES)
            cell=dict(phase=phase, factor=factor, statistics={}, differences={})
            for regime in REGIMES:
                cell['statistics'][regime]={metric:stats([values(r)[metric] for r in grouped[regime]])
                                           for metric in METRICS}
            for metric in METRICS:
                idle=np.array(cell['statistics']['idle'][metric]['samples'])
                active=np.array(cell['statistics']['contention'][metric]['samples'])
                d=active-idle
                item=stats(d)
                half={4:3.182446305284263,5:2.7764451051977987}[args.repetitions]*d.std(ddof=1)/np.sqrt(args.repetitions)
                item['block_difference_95ci']=[float(d.mean()-half),float(d.mean()+half)]
                item['mean_change_percent']=float(d.mean()/abs(idle.mean())*100) if idle.mean() else None
                item['max_abs_sample_change']=float(abs(d).max())
                cell['differences'][metric]=item
            cells.append(cell)
    args.output.mkdir(parents=True,exist_ok=True)
    aggregate={g:{m:stats([values(r)[m] for r in rows if r['regime']==g]) for m in METRICS} for g in REGIMES}
    worker=[r['worker_cpu_seconds']/r['runtime_seconds'] for r in rows if r['regime']=='contention']
    result=dict(cells=cells, equally_weighted_mixture=aggregate,
                safety=dict(trials=count,safe=count,repetitions=args.repetitions,worker_cpu_seconds_per_wall_second=stats(worker)),
                measurements_sha256=hashlib.sha256((args.input/'measurements.jsonl').read_bytes()).hexdigest(),
                runner_manifest=source_manifest,
                caveat=f'{args.repetitions} observations per cell; CI is unadjusted paired-block t interval (df={args.repetitions-1}), not a simultaneous 48-cell guarantee. Pooled distributions mix checkpoint/action conditions. This is conditional next-state comparison, not trajectory visitation or policy evaluation.')
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# CPU contention: conditional next-state and reward comparison','',result['caveat'],'',
           '## Runtime matrix','', '|State|Action|Idle mean ± std (s)|Idle CV %|Contention mean ± std (s)|Contention CV %|Change %|Block difference 95% CI (s)|',
           '|---|---|---|---|---|---|---|---|']
    for c in cells:
        a=c['statistics']['idle']['runtime_seconds'];b=c['statistics']['contention']['runtime_seconds'];d=c['differences']['runtime_seconds']
        lo,hi=d['block_difference_95ci']
        lines.append(f"|S{c['phase']}|{label(c['factor'])}|{a['mean']:.6f} ± {a['std']:.6f}|{a['cv_percent']:.3f}|{b['mean']:.6f} ± {b['std']:.6f}|{b['cv_percent']:.3f}|{d['mean_change_percent']:+.3f}|[{lo:+.6f}, {hi:+.6f}]|")
    lines += ['', '## Individual runtime measurements (seconds)', '', '|State|Action|Regime|'+'|'.join(f'r{i}' for i in range(args.repetitions))+'|','|'+'---|'*(3+args.repetitions)]
    for c in cells:
        for g in REGIMES:
            lines.append(f"|S{c['phase']}|{label(c['factor'])}|{g}|"+'|'.join(f'{v:.9f}' for v in c['statistics'][g]['runtime_seconds']['samples'])+'|')
    for metric in METRICS[1:]:
        lines += ['',f'## {metric}', '', '|State|Action|Idle mean ± std|Idle CV %|Contention mean ± std|Contention CV %|Difference|','|---|---|---|---|---|---|---|']
        for c in cells:
            a=c['statistics']['idle'][metric];b=c['statistics']['contention'][metric];d=c['differences'][metric]
            lines.append(f"|S{c['phase']}|{label(c['factor'])}|{a['mean']:.9g} ± {a['std']:.4g}|{a['cv_percent'] or 0:.4g}|{b['mean']:.9g} ± {b['std']:.4g}|{b['cv_percent'] or 0:.4g}|{d['mean']:+.6g}|")
    (args.output/'MATRIX.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharey=True)
    for phase,ax in enumerate(axes.flat,1):
        cs=[c for c in cells if c['phase']==phase]
        for j,c in enumerate(cs):
            for g,offset,color in [('idle',-.14,'tab:blue'),('contention',.14,'tab:orange')]:
                a=c['statistics'][g]['runtime_seconds']
                ax.scatter(np.repeat(j+offset,args.repetitions),a['samples'],s=18,alpha=.65,color=color)
                ax.errorbar(j+offset,a['mean'],yerr=a['std'],fmt='_',color=color,capsize=3)
        ax.set(title=f'S{phase}',xticks=range(6),xticklabels=[label(f) for f in FACTORS],ylabel='Action + 500 steps (s)')
        ax.grid(alpha=.2)
    fig.suptitle(f'Idle (blue) / contention (orange): {args.repetitions} independent processes per condition')
    fig.tight_layout();fig.savefig(args.output/'contention_runtimes.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,METRICS[2:5]):
        for c in cells:
            a=c['statistics']['idle'][metric];b=c['statistics']['contention'][metric]
            ax.scatter(a['mean'],b['mean'],s=22,alpha=.65)
        limits=ax.get_xlim();low=min(limits[0],ax.get_ylim()[0]);high=max(limits[1],ax.get_ylim()[1]);ax.plot([low,high],[low,high],'k--',lw=1)
        ax.set(title=metric,xlabel='Idle next-state mean',ylabel='Contention next-state mean')
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'contention_imbalance.png',dpi=180);plt.close(fig)
    pca_info={}
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,features,title in zip(axes,[METRICS[2:5],METRICS[2:]],['Imbalance only','Imbalance + timing per step']):
        idle=np.array([[values(r)[m] for m in features] for r in rows if r['regime']=='idle'])
        active=np.array([[values(r)[m] for m in features] for r in rows if r['regime']=='contention'])
        mu=idle.mean(0);scale=idle.std(0);z=(idle-mu)/scale
        _,s,v=np.linalg.svd(z,full_matrices=False);ratio=s*s/(s*s).sum()
        for array,name,color in [(idle,'idle','tab:blue'),(active,'contention','tab:orange')]:
            coords=((array-mu)/scale)@v[:2].T
            ax.scatter(coords[:,0],coords[:,1],s=40 if name=='idle' else 16,alpha=.55,
                       facecolors='none' if name=='idle' else color,edgecolors=color,label=name)
        ax.set(title=title,xlabel=f'PC1 ({ratio[0]:.1%} idle variance)',ylabel=f'PC2 ({ratio[1]:.1%})');ax.legend();ax.grid(alpha=.2)
        pca_info[title]=dict(features=features,fit_regime='idle',mean=mu.tolist(),scale=scale.tolist(),components=v[:2].tolist(),variance_ratio=ratio[:2].tolist())
    fig.tight_layout();fig.savefig(args.output/'contention_pca.png',dpi=180);plt.close(fig)
    (args.output/'pca_metadata.json').write_text(json.dumps(pca_info,indent=2)+'\n')
    print(json.dumps(dict(runtime_mixture=aggregate['contention']['runtime_seconds']['mean']/aggregate['idle']['runtime_seconds']['mean'],
                         max_imbalance_change={m:max(c['differences'][m]['max_abs_sample_change'] for c in cells) for m in METRICS[2:5]})))


if __name__=='__main__':
    main()
