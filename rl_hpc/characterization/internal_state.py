#!/usr/bin/env python3
"""Checkpoint-fork measurement of action response to LAMMPS internal state.

No co-runner is used.  A baseline trajectory creates binary LAMMPS restart
files.  Every measurement is then a fresh process reading the same restart,
changing only the absolute neighbour settings, and running 500 steps.
"""
from __future__ import annotations
import argparse, json, os, random, re, shlex, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from characterize import ROOT, git_commit, parse_log

GPU_LMP_LJ=ROOT/'build_kokkos_cuda/lmp'
GPU_LMP_SPCE=ROOT/'build_kokkos_cuda_char/lmp'  # includes MOLECULE + KSPACE

def lmp_for(workload: str) -> Path:
    return GPU_LMP_LJ if workload=='lj' else GPU_LMP_SPCE

def actions(workload: str):
    if workload=='lj': return [(s,e) for s in (.4,.6,.8,1.0,1.2) for e in (1,5,10,20)]
    # SPC/E uses real units; the LJ skin values are physically inapplicable.
    # The first checkpoint preflight excluded (2,10), (2,20), and (3,20)
    # because they produced dangerous neighbor builds.  Do not sample known
    # unsafe configurations in the main factorial.
    return [(2.0,1),(2.0,5),(3.0,1),(3.0,5),(3.0,10),(4.0,1),(4.0,5),(4.0,10),(4.0,20)]

def input_prefix(workload: str) -> list[str]:
    if workload=='lj': return [
        'units lj','atom_style atomic','lattice fcc 0.8442',
        'region box block 0 40 0 40 0 20','create_box 1 box','create_atoms 1 box','mass 1 1.0',
        'velocity all create 1.44 87287 loop geom','pair_style lj/cut 2.5','pair_coeff 1 1 1.0 1.0 2.5',
        'neighbor 0.6 bin','neigh_modify delay 0 every 5 check yes','fix integrator all nve',
        'thermo 500','thermo_style custom step temp press density pe ke etotal','timer normal']
    data=ROOT/'bench/POTENTIALS/data.spce'
    return [
        'units real','atom_style full',f'read_data {data}','replicate 2 4 1',
        'pair_style lj/cut/coul/long 9.8 9.8','kspace_style pppm 1.0e-4',
        'pair_coeff 1 1 0.15535 3.166','pair_coeff * 2 0.0000 0.0000',
        'bond_style harmonic','angle_style harmonic','dihedral_style none','improper_style none',
        'bond_coeff 1 1000.00 1.000','angle_coeff 1 100.0 109.47','special_bonds lj/coul 0.0 0.0 0.5',
        'neighbor 3.0 bin','neigh_modify delay 0 every 5 check yes',
        'fix shake all shake 0.0001 20 0 b 1 a 1','fix integrator all nvt temp 300.0 300.0 100.0',
        'velocity all create 300 432567 dist uniform','timestep 2.0',
        'thermo 500','thermo_style custom step temp press density pe ke etotal','timer normal']

def thermo(log: Path) -> dict[str,float]:
    lines=log.read_text(errors='replace').splitlines() if log.exists() else []
    keys=[]; latest={}
    for line in lines:
        fields=line.split()
        if fields and fields[0]=='Step': keys=fields; continue
        if keys and len(fields)==len(keys):
            try: latest={k:float(v) for k,v in zip(keys,fields)}
            except ValueError: pass
    return {'step':latest.get('Step',0.),'temperature':latest.get('Temp',0.),'pressure':latest.get('Press',0.),'density':latest.get('Density',0.),'pe':latest.get('PotEng',0.)}

def baseline_states(log: Path, checkpoints: int) -> list[dict[str,Any]]:
    """Read state at each baseline run boundary, before any fork is made."""
    lines=log.read_text(errors='replace').splitlines(); keys=[]; current={}; records=[]
    loop_indices=[]
    for i,line in enumerate(lines):
        fields=line.split()
        if fields and fields[0]=='Step': keys=fields
        elif keys and len(fields)==len(keys):
            try: current={k:float(v) for k,v in zip(keys,fields)}
            except ValueError: pass
        if line.startswith('Loop time of'):
            loop_indices.append((i,current.copy()))
    row_pattern=re.compile(r'^\s*(Pair|Bond|Neigh|Kspace|KSpace|Comm|Output|Modify|Other)\s+\|\s+[0-9.eE+-]+\s+\|\s+([0-9.eE+-]+)\s+\|')
    for ordinal,(start,state) in enumerate(loop_indices[:checkpoints]):
        end=loop_indices[ordinal+1][0] if ordinal+1<len(loop_indices) else len(lines)
        timing={}
        for line in lines[start:end]:
            match=row_pattern.match(line)
            if match: timing[match.group(1).lower()]=float(match.group(2))
        block='\n'.join(lines[start:end]); builds=re.findall(r'Neighbor list builds =\s*([0-9]+)',block)
        records.append({'checkpoint_id':f'S{ordinal+1}','simulation_step':state.get('Step',0.),'temperature':state.get('Temp',0.),'pressure':state.get('Press',0.),'density':state.get('Density',0.),'pe':state.get('PotEng',0.),'previous_segment_timing_seconds':timing,'neighbor_builds':int(builds[-1]) if builds else 0})
    return records

def command(workload: str, script: Path, log: Path) -> list[str]:
    return ['taskset','-c','0-3',str(lmp_for(workload)),'-k','on','g','1','-sf','kk','-pk','kokkos','gpu/aware','off','-in',str(script),'-log',str(log),'-screen','none']

def execute(workload: str, script: Path, log: Path, cwd: Path) -> tuple[int,float]:
    start=time.perf_counter(); p=subprocess.run(command(workload,script,log),cwd=cwd,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1'},stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=180); return p.returncode,time.perf_counter()-start

def baseline(out: Path, workload: str, checkpoints: int, interval: int) -> list[dict[str,Any]]:
    script=out/'baseline.in'; log=out/'baseline.log'; lines=input_prefix(workload)
    for i in range(checkpoints):
        lines += [f'run {interval}',f'write_restart {out / f"checkpoint-{i:02d}.restart"}',f'print "INTERNAL_CHECKPOINT {i+1}"']
    script.write_text('\n'.join(lines)+'\n')
    rc,wall=execute(workload,script,log,out)
    if rc: raise RuntimeError(f'baseline failed: {rc}')
    states=baseline_states(log,checkpoints)
    return [{**states[i],'restart':str(out/f'checkpoint-{i:02d}.restart'),'baseline_process_wall_seconds':wall} for i in range(checkpoints)]

def fork_input(workload: str, restart: Path, skin: float, every: int, steps: int) -> str:
    lines=[f'read_restart {restart}']
    # KSpace styles and time-integration fixes are intentionally not stored in
    # LAMMPS restart files, so restore them explicitly for an SPC/E fork.
    if workload=='spce':
        lines += ['kspace_style pppm 1.0e-4','fix shake all shake 0.0001 20 0 b 1 a 1','fix integrator all nvt temp 300.0 300.0 100.0']
    lines += [f'neighbor {skin} bin',f'neigh_modify delay 0 every {every} check yes','thermo 500','thermo_style custom step temp press density pe ke etotal','timer normal',f'run {steps}']
    return '\n'.join(lines)+'\n'

def trial(out: Path, workload: str, checkpoint: dict[str,Any], action: tuple[float,int], rep: int, steps: int) -> dict[str,Any]:
    skin,every=action; key=f"{checkpoint['checkpoint_id']}-s{skin:g}-e{every}-r{rep}"
    run=out/'runs'/key; run.mkdir(parents=True,exist_ok=True); script=run/'in.lammps';log=run/'log.lammps';script.write_text(fork_input(workload,Path(checkpoint['restart']),skin,every,steps))
    before=datetime.now(timezone.utc).isoformat(); rc,wall=execute(workload,script,log,run); metrics=parse_log(log); state=thermo(log)
    safe=rc==0 and metrics.get('dangerous_builds',0)==0 and all(abs(v)<1e100 for v in state.values())
    return {'timestamp':before,'git_commit':git_commit(),'workload':workload,'checkpoint_id':checkpoint['checkpoint_id'],'restart':checkpoint['restart'],'action':{'skin':skin,'every':every,'delay':0,'check':'yes'},'repetition':rep,'steps':steps,'process_wall_seconds':wall,'lammps':metrics,'thermo_after':state,'safe':safe,'returncode':rc,'command':shlex.join(command(workload,script,log))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--workload',choices=('lj','spce'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoints',type=int,default=5);p.add_argument('--interval',type=int,default=1000);p.add_argument('--steps',type=int,default=500);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--seed',type=int,default=20260923);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if not lmp_for(a.workload).exists():raise SystemExit(f'missing {lmp_for(a.workload)}')
 out=a.output if a.output.is_absolute() else ROOT/a.output
 if out.exists():p.error('output exists')
 out.mkdir(parents=True); checkpoint=baseline(out,a.workload,a.checkpoints,a.interval)
 state_records=[{k:v for k,v in cp.items() if k!='restart'} for cp in checkpoint]
 rng=random.Random(a.seed); todo=[]
 for cp in checkpoint[:1] if a.preflight_only else checkpoint:
  for action in actions(a.workload):
   for rep in range(1,a.repetitions+1):todo.append((cp,action,rep))
 rng.shuffle(todo); rows=[]
 for cp,action,rep in todo:
  row=trial(out,a.workload,cp,action,rep,a.steps);rows.append(row)
  print(json.dumps({'checkpoint':cp['checkpoint_id'],'action':action,'rep':rep,'safe':row['safe'],'loop':row['lammps'].get('loop_seconds')}),flush=True)
 (out/'measurements.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(out/'checkpoint_states.json').write_text(json.dumps(state_records,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'workload':a.workload,'checkpoints':a.checkpoints,'interval':a.interval,'steps':a.steps,'repetitions':a.repetitions,'preflight_only':a.preflight_only,'git_commit':git_commit()},indent=2)+'\n')
if __name__=='__main__':main()
