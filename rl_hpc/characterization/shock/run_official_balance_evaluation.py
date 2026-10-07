#!/usr/bin/env python3
"""Run/resume official fix-balance comparison, then analyze it."""
from pathlib import Path
import subprocess, sys

ROOT=Path(__file__).resolve().parents[3]
SHOCK=ROOT/'rl_hpc/characterization/shock'
DATA=ROOT/'rl_hpc/characterization/data'
OUT=DATA/'shock_official_balance_eval_main_v1'
RESTARTS=DATA/'shock_online_seed_restarts_v1/test_restart_list.json'

cmd=[sys.executable,'-u',str(SHOCK/'evaluate_official_balance.py'),
     '--output',str(OUT),'--restart-list',str(RESTARTS),'--decisions','120',
     '--repeats','2','--regimes','idle','jitter','--policies',
     'official_atoms','official_neigh_10','official_neigh_15','official_time_10']
if OUT.exists(): cmd.append('--resume')
subprocess.run(cmd,cwd=ROOT,check=True)
subprocess.run([sys.executable,str(SHOCK/'analyze_official_balance.py'),str(OUT)],cwd=ROOT,check=True)
