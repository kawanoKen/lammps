#!/usr/bin/env python3
"""Offline ridge fitted-Q iteration for the 25-decision shock pilot.

No simulator calls, no online updates. Checkpoint-fork measurements are NOT
treated as sequential transitions. Runtime prediction equals gamma=0 FQI.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ACTIONS = list(itertools.product((.25,.35,.5,.7,1.),(.5,.65,.8,1.,1.2,1.5)))
FIELDS = ['step','temperature','pressure','density','total_energy_per_atom',
          'shock_position','skin','factor','atom_imbalance']


def state(s, decision):
    p=s['previous_segment']; steps=p['steps']
    x=[float(s[k]) for k in FIELDS]+[(25-decision)/25]
    x += [p['loop_seconds']/steps, p['neighbor_builds']/steps]
    x += [p['timing_avg_seconds'].get(k,0)/steps for k in
          ['pair','neigh','comm','modify','other']]
    x += [p[k][v] for k in ['nlocal','nghost','neighs'] for v in ['mean','max','min']]
    # The saved balance mesh carries the persistent spatial partition.
    vertices=[]
    for line in s['partition_mesh'].splitlines():
        t=line.split()
        if len(t)==5:
            try: vertices.append(float(t[2]))
            except ValueError: pass
    assert len(vertices)==256, 'Expected 32 brick domains with 8 vertices each'
    x += [max(vertices[i:i+8])-min(vertices[i:i+8]) for i in range(0,256,8)]
    return x


def features(z, actions):
    a=np.asarray(actions,float)
    u=(a[:,0]-.625)/.375; v=(a[:,1]-1)/.5
    av=np.column_stack([u,v,u*u,v*v,u*v])
    return np.column_stack([np.ones(len(z)),z,av,(z[:,:,None]*av[:,None,:]).reshape(len(z),-1)])


def candidates(z):
    return features(np.repeat(z,len(ACTIONS),axis=0),np.tile(ACTIONS,(len(z),1))).reshape(len(z),len(ACTIONS),-1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[ROOT/'rl_hpc/characterization/data/shock_trajectory_background_20260927/pilot',
             ROOT/'rl_hpc/characterization/data/shock_trajectory_coverage_20260927']
    rows=[]; hashes={}; splits={k:[] for k in ['train','validation','test']}
    for source_index,source in enumerate(sources):
        files=sorted(source.glob('episode-*/transitions.jsonl'))
        random.Random(928+source_index).shuffle(files)
        counts=(2,0) if source_index==0 else (13,4)
        for i,f in enumerate(files):
            eid=f'{source_index}/{f.parent.name}'
            part='train' if i<counts[0] else 'validation' if i<sum(counts) else 'test'
            splits[part].append(eid); hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
            episode=[json.loads(l) for l in f.read_text().splitlines()]
            assert len(episode)==25
            for j,r in enumerate(episode):
                assert r['safe'] and r['runtime']>0 and r['reward']==-r['runtime']
                assert r['next_state']['step']==r['state']['step']+500
                assert not j or episode[j-1]['next_state']==r['state']
                a=[r['action']['skin'],r['action']['factor']]
                expected=r['epsilon']/30+(1-r['epsilon'] if a==r['rule_action'] else 0)
                assert abs(expected-r['behavior_probability'])<1e-12
                assert abs(r['runtime']-sum(r[k] for k in ['preparation_seconds','balance_seconds','run_seconds']))<1e-6
                assert r['next_state']['previous_segment']['dangerous_builds']==0
                rows.append(dict(r,episode_id=eid,split=part))
    X=np.asarray([state(r['state'],r['decision']) for r in rows]); NX=np.asarray([state(r['next_state'],r['decision']+1) for r in rows])
    assert np.isfinite(X).all() and np.isfinite(NX).all()
    actions=np.asarray([[r['action']['skin'],r['action']['factor']] for r in rows])
    rewards=-np.asarray([r['runtime'] for r in rows]); remaining=np.asarray([24-r['decision'] for r in rows])
    masks={k:np.asarray([r['split']==k for r in rows]) for k in splits}
    mean=X[masks['train']].mean(0); scale=X[masks['train']].std(0);scale[scale<1e-8]=1
    z=(X-mean)/scale; nz=(NX-mean)/scale
    F=features(z,actions); C=candidates(nz); current=candidates(z)
    coverage={}
    for part in ['all','train','validation','test']:
        counts=np.zeros((5,30),int)
        for r in rows:
            if part=='all' or r['split']==part:
                counts[r['decision']//5,ACTIONS.index((r['action']['skin'],r['action']['factor']))]+=1
        coverage[part]={'counts':counts.tolist(),'visited_per_phase':(counts>0).sum(1).tolist(),'missing_phase_actions':int((counts==0).sum())}
    manifest=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  splits=splits,actions=ACTIONS,transitions=len(rows),coverage=coverage,
                  features='FIELDS, remaining fraction, per-step previous timings/builds, rank statistics, 32 domain widths',
                  fields=FIELDS,finite_horizon=25,normalization='train-only',seeds=[11,22,33],
                  caveat='one physical initial condition; summarized state may be partially observed; no live policy evaluation')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    metrics=[]
    for seed in [11,22,33]:
        rng=np.random.default_rng(seed)
        # Whole training-episode bootstrap, never individual-transition split.
        sampled=rng.choice(splits['train'],len(splits['train']),replace=True)
        idx=np.concatenate([np.flatnonzero([r['episode_id']==e for r in rows]) for e in sampled])
        f=F[idx]; penalties=np.eye(f.shape[1]);penalties[0,0]=0
        best=None
        for ridge in [1.,10.,100.]:
            solver=np.linalg.solve(f.T@f+ridge*penalties,f.T)
            w=solver@rewards[idx]
            val=float(np.mean((F[masks['validation']]@w-rewards[masks['validation']])**2))
            if best is None or val<best[0]:best=(val,ridge,solver)
        _,ridge,solver=best
        for gamma in [0.,.97]:
            w=np.zeros(F.shape[1]); history=[]
            # Fixed iteration budget, NOT selected on test performance.
            for iteration in range(1 if gamma==0 else 25):
                future=(C[idx]@w).max(1)
                lower=rewards[masks['train']].min()*remaining[idx]
                target=rewards[idx]+gamma*np.clip(future,lower,0)*(remaining[idx]>0)
                w=solver@target
                assert np.isfinite(w).all()
                history.append(float(np.mean((f@w-target)**2)))
            label='predictor_myopic' if gamma==0 else 'fqi_sequential'
            np.savez(args.output/f'{label}_seed{seed}.npz',coef=w,mean=mean,scale=scale,actions=np.asarray(ACTIONS))
            test=masks['test']; picks=(current[test]@w).argmax(1)
            metric=dict(method=label,seed=seed,gamma=gamma,ridge=ridge,train_loss=history,
                        predicted_q_range=[float((F[test]@w).min()),float((F[test]@w).max())],
                        selected_action_counts=np.bincount(picks,minlength=30).tolist())
            if gamma==0:
                error=F[test]@w-rewards[test]
                metric.update(test_runtime_rmse_seconds=float(np.sqrt(np.mean(error**2))),test_runtime_mae_seconds=float(np.mean(abs(error))),validation_rmse_seconds=float(np.sqrt(best[0])))
            metrics.append(metric)
            print(label,seed,'ridge',ridge,'test runtime MAE',metric.get('test_runtime_mae_seconds','not a runtime predictor'),flush=True)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print('COMPLETE',len(rows),'transitions',coverage['train']['visited_per_phase'],flush=True)


if __name__=='__main__':main()
