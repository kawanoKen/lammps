#!/usr/bin/env python3
"""Execute one frozen policy live against a recorded hidden regime schedule."""
from __future__ import annotations
import argparse, json, os, random, signal, subprocess, time
from pathlib import Path
from env import LammpsGpuEnv, ROOT
from models import LinearQ, valid_actions

SPECS={'idle':None,'light':(64,8,1000),'medium_low':(128,16,500),'medium_high':(128,32,0),'heavy':(256,64,0)}
ORACLE_TARGETS={'idle':(.6,5),'light':(.6,5),'medium_low':(.8,5),'medium_high':(1.0,10),'heavy':(1.0,10)}

def toward(env, target):
    skin,every=target
    ds=max(-.2,min(.2,skin-env.skin))
    valid=[de for de in (-5,-1,0,1,5) if 1 <= env.every + de <= 10]
    # Respect the discrete delta constraint and choose the reachable value
    # closest to the desired absolute setting (no one-step teleportation).
    de=min(valid,key=lambda x:abs((env.every+x)-every))
    return (ds,de)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--policy',choices=('model','fixed','heuristic','oracle','random'),required=True); p.add_argument('--schedule',required=True); p.add_argument('--initial-skin',type=float,required=True); p.add_argument('--initial-every',type=int,required=True); p.add_argument('--model',type=Path); p.add_argument('--fixed-skin',type=float,default=.6); p.add_argument('--fixed-every',type=int,default=5); p.add_argument('--seed',type=int,default=1); a=p.parse_args()
    schedule=a.schedule.split(',')
    if not schedule or any(x not in SPECS for x in schedule): raise SystemExit('invalid schedule')
    if a.policy=='model' and not a.model: raise SystemExit('--model required')
    model=LinearQ.load(a.model) if a.policy=='model' else None; rng=random.Random(a.seed)
    bg=None; active=None
    def cleanup(*_):
        nonlocal bg
        if bg and bg.poll() is None: os.killpg(bg.pid,signal.SIGTERM)
        os._exit(0)
    signal.signal(signal.SIGTERM,cleanup);signal.signal(signal.SIGINT,cleanup)
    def regime(name):
        nonlocal bg,active
        if name==active:return
        if bg and bg.poll() is None: os.killpg(bg.pid,signal.SIGTERM);bg.wait(timeout=8)
        bg=None;active=name
        if SPECS[name]:
            mb,reps,sleep=SPECS[name]; binary=ROOT/'rl_hpc/characterization/build/gpu_contender'
            bg=subprocess.Popen(['taskset','-c','8-11',str(binary),'180',str(mb),str(reps),str(sleep)],env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},start_new_session=True);time.sleep(.5)
    initial=(a.fixed_skin,a.fixed_every) if a.policy=='fixed' else (a.initial_skin,a.initial_every)
    regime(schedule[0]); env=LammpsGpuEnv(); obs=env.reset(*initial); rows=[]
    for t,label in enumerate(schedule):
        regime(label)
        if a.policy=='fixed': action=(0.0,0)
        elif a.policy=='oracle': action=toward(env,ORACLE_TARGETS[label])
        elif a.policy=='heuristic':
            gpu=obs.get('gpu',{}); target=(1.0,10) if float(gpu.get('util',0)) >= 50 or float(gpu.get('power',0)) >= 120 else (.6,5)
            action=toward(env,target)
        elif a.policy=='random': action=rng.choice(valid_actions(obs, skin_points=21))
        else:
            ds,de,_=model.best_action(obs);action=(ds,de)
        before=time.perf_counter(); nxt,reward,terminated,info=env.step(action); overhead=time.perf_counter()-before+reward
        rows.append({'episode_id':a.output.stem,'decision':t,'policy':a.policy,'model':str(a.model) if a.model else None,'contention_ground_truth':label,'state_before':obs,'requested_action':action,'applied_action':info['applied_action'],'configuration':info['configuration'],'segment_runtime_seconds':-reward,'action_application_overhead_seconds':overhead,'state_after':nxt,'unsafe':info['unsafe'],'power_cap_flag':bool(nxt.get('gpu',{}).get('power_limit',0) and nxt['gpu'].get('power',0)>=.98*nxt['gpu']['power_limit'])})
        obs=nxt
        if terminated: break
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(x,default=str)+'\n' for x in rows))
    if bg and bg.poll() is None: os.killpg(bg.pid,signal.SIGTERM)
    os._exit(0)
if __name__=='__main__':main()
