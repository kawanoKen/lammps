#!/usr/bin/env python3
"""Persistent MPI32 shock pilot, observable heuristic plus uniform exploration."""
import argparse
import collections
import itertools
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import signal
import subprocess
import threading
import time

import psutil

import numeric_sweep as sweep
import shock_counterfactual as base

ACTIONS=list(itertools.product(sweep.SKINS,sweep.FACTORS))


class Simulation:
    def __init__(self, directory):
        self.directory=directory
        self.command=['mpirun','--nooversubscribe','--bind-to','core','--map-by','core',
                      '-np','32',str(base.LMP),'-nonbuf','-echo','none','-log',str(directory/'log.lammps')]
        self.stderr=(directory/'stderr.txt').open('w')
        self.process=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=self.stderr,text=True,bufsize=1,start_new_session=True,
                                      cwd=directory,env={**os.environ,'OMP_NUM_THREADS':'1'})
        self.queue=queue.Queue()
        self.index=0
        def pump():
            for line in self.process.stdout: self.queue.put(line)
            self.queue.put(None)
        self.reader=threading.Thread(target=pump,daemon=True);self.reader.start()

    def execute(self, commands):
        marker=f'TRAJECTORY_DONE_{self.index}';self.index+=1
        self.process.stdin.write(commands+f'\nprint "{marker}"\n');self.process.stdin.flush()
        lines=[];deadline=time.monotonic()+180
        while True:
            line=self.queue.get(timeout=max(.01,deadline-time.monotonic()))
            if line is None: raise RuntimeError('LAMMPS exited: '+''.join(lines)[-2000:])
            if line.strip()==marker: return ''.join(lines)
            lines.append(line)

    def close(self, graceful=False):
        try:
            if graceful and self.process.poll() is None:
                try:
                    self.process.stdin.write('quit\n');self.process.stdin.flush()
                    self.process.wait(timeout=15)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    pass
            if self.process.poll() is None:
                try:
                    descendants=psutil.Process(self.process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    descendants=[]
                try:os.killpg(self.process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                for child in descendants:
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                try:self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    try:os.killpg(self.process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    self.process.wait()
                for child in descendants:
                    try:
                        if child.is_running():child.kill()
                    except psutil.NoSuchProcess:pass
        finally:self.stderr.close()
        return self.process.returncode


def observation(text,skin,factor,expected_atoms=138240):
    blocks=base.timing_blocks(text)
    metric=next((b for b in reversed(blocks) if b['steps']>0),{})
    matches=re.findall(r'^OBS (.*)$',text,re.M)
    values=list(map(float,matches[-1].split())) if matches else []
    if len(values)!=7 or not all(math.isfinite(v) for v in values):
        raise RuntimeError('Missing/nonfinite thermo')
    step,atoms,temp,pressure,density,energy,front=values
    if atoms!=expected_atoms or temp<=0 or density<=0:raise RuntimeError('Invalid physics/atom count')
    builds=re.findall(r'Dangerous builds =\s*(\d+)',text)
    if not builds or any(int(b)>0 for b in builds):raise RuntimeError('Dangerous or missing build count')
    if 'weight neigh skipped' in text or 'ERROR' in text:raise RuntimeError('Invalid balance execution')
    nlocal=metric.get('nlocal')
    if not nlocal:raise RuntimeError('Missing local atom statistics')
    return dict(step=step,atoms=atoms,temperature=temp,pressure=pressure,density=density,
                total_energy_per_atom=energy,shock_position=front,skin=skin,factor=factor,
                atom_imbalance=nlocal['max']/nlocal['mean'],previous_segment=metric)


OBS='print "OBS $(step) $(atoms) $(temp) $(press) $(density) $(etotal) $(v_xshock)"'


def rule(state):
    # Explicit provisional behavior rule, not a fitted optimum or an oracle.
    # Respond only to pre-decision observations, never checkpoint labels.
    return (.5,1.5) if state['atom_imbalance']>1.2 else (.35,.8)


def episode(out,index,decisions,epsilon,seed,restart,expected_atoms):
    directory=out/f'episode-{index:03d}';directory.mkdir()
    rng=random.Random(seed)
    initial=rng.choice(ACTIONS)
    simulation=Simulation(directory);started=time.time();status='failed'
    try:
        setup=base.fork_input(restart,*[initial[0],'none',500,50]).split('variable action_start timer')[0]
        # Initial factor denotes the first actual partitioning operation.
        setup+=f'\nneighbor {initial[0]} bin\nrun 0 post no\nbalance 1.0 shift x 10 1.0 weight neigh {initial[1]} out initial.mesh\nrun 50\n'+OBS
        text=simulation.execute(setup)
        state=observation(text,*initial,expected_atoms=expected_atoms)
        state['partition_mesh']=(directory/'initial.mesh').read_text()
        start_step=state['step']
        with (directory/'transitions.jsonl').open('w') as stream:
            for t in range(decisions):
                recommended=rule(state)
                exploratory=rng.random()<epsilon
                action=rng.choice(ACTIONS) if exploratory else recommended
                probability=epsilon/len(ACTIONS)+(1-epsilon if action==recommended else 0)
                skin,factor=action
                commands=f'''variable begin timer
neighbor {skin} bin
neigh_modify every 1 delay 0 check yes
run 0 post no
variable prepared timer
balance 1.0 shift x 10 1.0 weight neigh {factor} out partition-{t:03d}.mesh
variable balanced timer
run 500
variable ended timer
print "TIMES $(v_ended-v_begin:%.9f) $(v_prepared-v_begin:%.9f) $(v_balanced-v_prepared:%.9f) $(v_ended-v_balanced:%.9f)"
{OBS}
'''
                wall=time.monotonic()
                text=simulation.execute(commands)
                elapsed=time.monotonic()-wall
                (directory/f'segment-{t:03d}.txt').write_text(text)
                next_state=observation(text,skin,factor,expected_atoms=expected_atoms)
                if next_state['step']!=state['step']+500:raise RuntimeError('Non-500-step transition')
                next_state['partition_mesh']=(directory/f'partition-{t:03d}.mesh').read_text()
                times=list(map(float,re.findall(r'^TIMES (.*)$',text,re.M)[-1].split()))
                if not all(math.isfinite(v) and v>=0 for v in times) or abs(times[0]-sum(times[1:]))>1e-6:
                    raise RuntimeError('Invalid timer identity')
                row=dict(dataset_version='shock-trajectory-v2',episode=index,decision=t,
                         timestamp=time.time(),behavior_seed=seed,mpi_ranks=32,omp_threads=1,
                         state=state,action=dict(skin=skin,factor=factor),rule_action=recommended,
                         exploratory=exploratory,behavior_probability=probability,epsilon=epsilon,
                         reward=-times[0],runtime=times[0],preparation_seconds=times[1],
                         balance_seconds=times[2],run_seconds=times[3],roundtrip_seconds=elapsed,
                         next_state=next_state,terminated=False,truncated=t==decisions-1,safe=True)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                print(index,t,action,'explore',exploratory,'seconds',times[0],flush=True)
                state=next_state
        rc=simulation.close(graceful=True)
        if rc!=0:raise RuntimeError(f'Exit {rc}')
        status='complete'
        return dict(episode=index,decisions=decisions,start_step=start_step,end_step=state['step'],
                    process_exit=rc,wall_seconds=time.time()-started,initial_action=initial)
    except BaseException as exc:
        (directory/'failure.json').write_text(json.dumps(dict(error=str(exc),time=time.time())))
        raise
    finally:
        simulation.close()
        (directory/'status.json').write_text(json.dumps(dict(status=status,time=time.time())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=3);p.add_argument('--decisions',type=int,default=24)
    p.add_argument('--epsilon',type=float,default=.4)
    p.add_argument('--seed-base',type=int,default=20260927)
    p.add_argument('--restart',type=Path,default=sweep.SOURCE/'checkpoint-S1.restart')
    p.add_argument('--expected-atoms',type=int,default=138240)
    a=p.parse_args()
    if not 0<a.epsilon<=1 or not 1<=a.decisions<=50 or a.episodes<1:
        p.error('epsilon in (0,1], decisions in [1,50], episodes >= 1')
    restart=a.restart.resolve()
    if not restart.is_file():p.error('restart does not exist')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    manifest=dict(git_commit=base.git_commit(),binary_sha256=sweep.digest(base.LMP),
                  restart=str(restart),restart_sha256=sweep.digest(restart),expected_atoms=a.expected_atoms,
                  runner_sha256=sweep.digest(Path(__file__)),actions=ACTIONS,epsilon=a.epsilon,
                  seed_base=a.seed_base,mpi=32,omp=1,steps=500,decisions=a.decisions,
                  rule='imbalance>1.2: (0.5,1.5); otherwise (0.35,0.8)',
                  scope='one physical seed; varied behavior seeds only; no generalization claim',
                  diagnostics='CVCF/history omitted consistently',initialization='50 warmup, balance, 50 observation steps')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summaries=[episode(out,i,a.decisions,a.epsilon,a.seed_base+i,restart,a.expected_atoms) for i in range(a.episodes)]
    rows=[json.loads(l) for f in sorted(out.glob('episode-*/transitions.jsonl')) for l in f.read_text().splitlines()]
    counts=collections.Counter((r['action']['skin'],r['action']['factor']) for r in rows)
    report=dict(episodes=summaries,transitions=len(rows),exploration_count=sum(r['exploratory'] for r in rows),
                visited_actions=len(counts),action_counts={str(k):v for k,v in counts.items()},
                all_safe=all(r['safe'] for r in rows))
    (out/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
