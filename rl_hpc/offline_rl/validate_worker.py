#!/usr/bin/env python3
"""Launch one episode worker and prove that GPU 0 has no residual process."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

def processes():
 q='gpu_uuid,pid,process_name,used_memory'
 try: out=subprocess.check_output(['nvidia-smi',f'--query-compute-apps={q}','--format=csv,noheader'],text=True,timeout=5)
 except subprocess.CalledProcessError: return []
 return [x for x in out.splitlines() if x.strip() and 'No running' not in x]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--decisions',type=int,default=3);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 env=os.environ.copy();env['LD_PRELOAD']='/usr/lib/x86_64-linux-gnu/libstdc++.so.6';env.pop('LD_LIBRARY_PATH',None)
 cmd=['/usr/bin/python3',str(Path(__file__).with_name('episode_worker.py')),'--output',str(a.output),'--decisions',str(a.decisions)]
 result=subprocess.run(cmd,env=env,check=False,timeout=180)
 rows=[json.loads(x) for x in a.output.read_text().splitlines() if x]
 residual=processes(); summary={'worker_exit':result.returncode,'transitions':len(rows),'unsafe':sum(r['unsafe'] for r in rows),'residual_gpu_processes':residual,'ok':result.returncode==0 and bool(rows) and not residual}
 print(json.dumps(summary));raise SystemExit(0 if summary['ok'] else 1)
if __name__=='__main__':main()
