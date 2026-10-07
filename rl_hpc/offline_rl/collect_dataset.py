#!/usr/bin/env python3
"""Collect small episode-split offline data with hidden piecewise GPU regimes."""
from __future__ import annotations
import argparse,json,os,random,subprocess,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def schedule(rng,n):
 out=[]; levels=('idle','light','medium_low','medium_high','heavy')
 while len(out)<n: out += [rng.choice(levels)]*rng.randint(2,5)
 return out[:n]
def cool(temp,power,timeout=900):
 end=time.monotonic()+timeout
 while True:
  out=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,power.draw,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  row=next((x for x in out.splitlines() if x.split(',')[0].strip()=='0'),None)
  if row:
   _,t,p,u=(float(x.strip()) for x in row.split(','))
   if t<=temp and p<=power and u<=2:return {'temperature_c':t,'power_w':p,'utilization_percent':u}
  if time.monotonic()>end:raise RuntimeError('GPU 0 did not return to baseline')
  time.sleep(5)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=4);p.add_argument('--decisions',type=int,default=10);p.add_argument('--seed',type=int,default=7);p.add_argument('--cool-temp',type=float,default=45);p.add_argument('--cool-power',type=float,default=45);a=p.parse_args()
 out=a.output if a.output.is_absolute() else ROOT/a.output
 if out.exists():p.error('output exists')
 out.mkdir(parents=True); rng=random.Random(a.seed); merged=[]
 env=os.environ.copy();env['LD_PRELOAD']='/usr/lib/x86_64-linux-gnu/libstdc++.so.6';env.pop('LD_LIBRARY_PATH',None)
 for ep in range(a.episodes):
  baseline=cool(a.cool_temp,a.cool_power)
  sch=schedule(rng,a.decisions); path=out/f'episode-{ep:03d}.jsonl'
  cmd=['/usr/bin/python3',str(Path(__file__).with_name('episode_worker.py')),'--output',str(path),'--decisions',str(a.decisions),'--seed',str(rng.randrange(2**31)),'--schedule',','.join(sch)]
  proc=subprocess.Popen(cmd,env=env,start_new_session=True)
  try:r=proc.wait(timeout=300)
  except KeyboardInterrupt:
   os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=15);raise
  rows=[json.loads(x) for x in path.read_text().splitlines() if x]
  for row in rows:row['episode_baseline_gpu']=baseline
  path.write_text(''.join(json.dumps(x)+'\n' for x in rows)); merged+=rows
  if r or len(rows)!=a.decisions or any(x['unsafe'] for x in rows):raise SystemExit(f'episode {ep} invalid')
  print(json.dumps({'episode':ep,'transitions_collected':len(merged),'baseline_gpu':baseline}),flush=True)
 (out/'transitions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in merged));(out/'manifest.json').write_text(json.dumps({'episodes':a.episodes,'decisions':a.decisions,'transitions':len(merged),'seed':a.seed,'cool_temp':a.cool_temp,'cool_power':a.cool_power},indent=2)+'\n')
 print(json.dumps({'output':str(out),'transitions':len(merged)}))
if __name__=='__main__':main()
