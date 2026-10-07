#!/usr/bin/env python3
"""Run one persistent CUDA/LAMMPS episode, then exit without CUDA unload."""
from __future__ import annotations
import argparse, json, os, random, signal, subprocess
from pathlib import Path
from env import LammpsGpuEnv, ROOT
from pathlib import Path as P

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--decisions',type=int,default=3); p.add_argument('--seed',type=int,default=1); p.add_argument('--regime',default='idle'); p.add_argument('--contender',action='store_true'); p.add_argument('--schedule',default=''); p.add_argument('--initial-skin',type=float); p.add_argument('--initial-every',type=int); p.add_argument('--zero-action',action='store_true',help='preflight only: hold the initial configuration'); a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True); rng=random.Random(a.seed)
    # Avoid known unsafe low-skin / high-every cells until the full preflight.
    specs={'idle':None,'light':(64,8,1000),'medium_low':(128,16,500),'medium_high':(128,32,0),'heavy':(256,64,0)}
    schedule=a.schedule.split(',') if a.schedule else [a.regime]*a.decisions
    if len(schedule)!=a.decisions or any(x not in specs for x in schedule): raise SystemExit('schedule must contain one known regime per decision')
    bg=None; active=None
    def interrupted(signum, frame):
        del frame
        if bg and bg.poll() is None: os.killpg(bg.pid,signal.SIGTERM)
        os._exit(128+signum)
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    def set_regime(name):
        nonlocal bg,active
        if name==active:return
        if bg and bg.poll() is None: os.killpg(bg.pid,signal.SIGTERM); bg.wait(timeout=8)
        bg=None; active=name
        if specs[name]:
            mb,reps,sleep=specs[name]; binary=ROOT/'rl_hpc/characterization/build/gpu_contender'
            bg=subprocess.Popen(['taskset','-c','8-11',str(binary),'180',str(mb),str(reps),str(sleep)],env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},start_new_session=True); __import__('time').sleep(.5)
    set_regime(schedule[0])
    env=LammpsGpuEnv()
    initial_skin=a.initial_skin if a.initial_skin is not None else rng.choice((.6,.8,1.0))
    initial_every=a.initial_every if a.initial_every is not None else rng.choice((1,5,10))
    obs=env.reset(skin=initial_skin, every=initial_every)
    with a.output.open('w',encoding='utf-8') as f:
        for t in range(a.decisions):
            set_regime(schedule[t])
            # Sample only deltas that remain in the validated domain. This
            # preserves hybrid-action coverage without a boundary random walk.
            ds_lo=max(-.2,.6-env.skin); ds_hi=min(.2,1.2-env.skin)
            valid_every=[d for d in (-5,-1,0,1,5) if 1<=env.every+d<=10]
            if a.zero_action:
                requested=(0.0,0); prob=1.0
            else:
                requested=(rng.uniform(ds_lo,ds_hi),rng.choice(valid_every)); prob=(1/(ds_hi-ds_lo))*(1/len(valid_every))
            nxt,reward,terminated,info=env.step(requested)
            gpu=nxt.get('gpu',{}); limit=float(gpu.get('power_limit',0.0)); power=float(gpu.get('power',0.0))
            row={'dataset_version':'offline-rl-main-v2','episode_id':a.output.stem,'decision':t,'contention_ground_truth':schedule[t],'contention_config':specs[schedule[t]],'state_before':obs,'requested_action':requested,'applied_action':info['applied_action'],'configuration':info['configuration'],'segment_runtime_seconds':-reward,'reward':reward,'state_after':nxt,'unsafe':info['unsafe'],'terminated':terminated,'power_cap_flag':bool(limit and power >= 0.98*limit),'behavior_probability':prob}
            f.write(json.dumps(row,default=str)+'\n'); f.flush(); os.fsync(f.fileno())
            obs=nxt
            if terminated: break
    if bg and bg.poll() is None: os.killpg(bg.pid,signal.SIGTERM)
    # Kokkos CUDA currently aborts if CPython unloads liblammps. Process exit
    # releases the CUDA context while deliberately bypassing that destructor.
    os._exit(0)
if __name__=='__main__': main()
