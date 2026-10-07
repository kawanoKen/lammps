#!/usr/bin/env python3
"""Paired live evaluation of frozen offline policies on held-out episodes."""
from __future__ import annotations
import argparse, json, os, random, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def cool(temp: float=55., power: float=80., timeout: float=900.) -> dict:
    deadline=time.monotonic()+timeout
    while True:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,power.draw,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        line=next(x for x in out.splitlines() if x.split(',')[0].strip()=='0')
        _,t,p,u=(float(x.strip()) for x in line.split(','))
        if t<=temp and p<=power and u<=2:return {'temperature_c':t,'power_w':p,'utilization_percent':u}
        if time.monotonic()>deadline:raise RuntimeError('GPU 0 did not cool to the evaluation baseline')
        time.sleep(5)

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=10);p.add_argument('--seed',type=int,default=20260929);p.add_argument('--force-regime',choices=('idle','light','medium_low','medium_high','heavy'));a=p.parse_args()
    rows=[json.loads(x) for x in (a.dataset/'transitions.jsonl').read_text().splitlines() if x]
    split=json.loads((a.models/'split.json').read_text()); test=split['test'][:a.episodes]
    by_ep={e:[r for r in rows if r['episode_id']==e] for e in test}
    methods=[('fixed',None),('oracle',None),('heuristic',None),('random',None),('predictor',a.models/'predictor.json'),('fqi_myopic',a.models/'fqi_myopic.json'),('fqi_sequential',a.models/'fqi_sequential.json')]
    if a.output.exists():p.error('output exists')
    a.output.mkdir(parents=True); rng=random.Random(a.seed); all_rows=[]
    env=os.environ.copy();env['LD_PRELOAD']='/usr/lib/x86_64-linux-gnu/libstdc++.so.6';env.pop('LD_LIBRARY_PATH',None)
    for episode in test:
        sample=by_ep[episode]
        schedule=','.join(([a.force_regime]*len(sample)) if a.force_regime else [r['contention_ground_truth'] for r in sample])
        initial=sample[0]['state_before']
        order=methods[:];rng.shuffle(order)
        for method,model in order:
            baseline=cool(); path=a.output/f'{episode}-{method}.jsonl'
            cmd=['/usr/bin/python3',str(Path(__file__).with_name('policy_worker.py')),'--output',str(path),'--policy','model' if model else method,'--schedule',schedule,'--initial-skin',str(initial['skin']),'--initial-every',str(initial['every']),'--seed',str(a.seed)]
            if model:cmd += ['--model',str(model)]
            result=subprocess.run(cmd,env=env,timeout=300)
            data=[json.loads(x) for x in path.read_text().splitlines() if x]
            if result.returncode or len(data)!=len(sample) or any(r['unsafe'] for r in data):raise RuntimeError(f'{episode}/{method} invalid')
            for r in data:r['evaluation_method']=method;r['evaluation_baseline_gpu']=baseline
            path.write_text(''.join(json.dumps(r)+'\n' for r in data));all_rows+=data
            print(json.dumps({'episode':episode,'method':method,'segments':len(data),'seconds':sum(x['segment_runtime_seconds'] for x in data)}),flush=True)
    (a.output/'live_transitions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
    summary={}
    for method,_ in methods:
        x=[r for r in all_rows if r['evaluation_method']==method]
        summary[method]={'episodes':len({r['episode_id'] for r in x}),'seconds_total':sum(r['segment_runtime_seconds'] for r in x),'seconds_mean_episode':sum(r['segment_runtime_seconds'] for r in x)/len({r['episode_id'] for r in x}),'unsafe':sum(r['unsafe'] for r in x)}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
