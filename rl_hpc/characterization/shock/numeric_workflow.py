#!/usr/bin/env python3
"""Run discovery then frozen, independent confirmation; resumable on disk."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import analyze_numeric as analysis
import numeric_sweep as sweep


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'workflow.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'workflow.pid').write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM,sweep.stop);signal.signal(signal.SIGINT,sweep.stop)
    started=time.time()
    proc=None
    try:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('numeric_sweep.py')),
                               '--output',str(out),'--phase','main'])
        if proc.wait()!=0: raise RuntimeError('Discovery stopped; inspect progress.log')
        plan=analysis.analyze(out)
        lock2=(out/'lock').open('w');fcntl.flock(lock2,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cps={c['checkpoint_id']:c for c in json.loads((sweep.SOURCE/'checkpoint_states.json').read_text())}
        data=out/'measurements.jsonl'
        rows=[json.loads(l) for l in data.read_text().splitlines()]
        if any(not r['safe'] for r in rows): raise RuntimeError('Unsafe records')
        done={r['key'] for r in rows}
        with data.open('a') as stream:
            for job in plan:
                ph,ck,s,f,r=job;key=f'{ph}-{ck}-s{s:g}-w{f:g}-r{r}'
                if key in done: continue
                row=sweep.trial(out,job,cps)
                stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                print(key,'safe=',row['safe'],'seconds=',row['times']['ACTION_WALL_SECONDS'],flush=True)
                if not row['safe']: raise RuntimeError('Unsafe confirmation')
        analysis.analyze(out)
        (out/'workflow_complete.json').write_text(json.dumps(dict(completed=time.time(),wall_seconds=time.time()-started)))
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=20)
        sweep.clean_child()


if __name__=='__main__':main()
