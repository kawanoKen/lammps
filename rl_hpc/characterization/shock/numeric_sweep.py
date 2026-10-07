#!/usr/bin/env python3
"""Resumable, isolated 32-rank skin x neighbor-weight checkpoint experiment."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

import shock_counterfactual as base

SKINS = (.25, .35, .5, .7, 1.)
FACTORS = (.5, .65, .8, 1., 1.2, 1.5)
ROOT = base.ROOT
SOURCE = ROOT / 'rl_hpc/characterization/data/shock_nemd_main_20260926'
CHILD = None


def stop(signum, frame):
    raise KeyboardInterrupt


def clean_child():
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
        try:
            CHILD.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGKILL)
            CHILD.wait()
    CHILD = None


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def script(cp, skin, factor):
    # Preserve the common physical warmup and NVE/wall physics from the prior
    # forks. CVCF/history diagnostics remain excluded, explicitly in manifest.
    text = base.fork_input(Path(cp['restart']), skin, 'neigh', 500, 50)
    text = text.replace('weight neigh 0.8', f'weight neigh {factor:g}')
    text = text.replace('thermo_style custom step temp pe density etotal press v_xshock',
                        'thermo_style custom step atoms temp pe density etotal press v_xshock')
    text = text.replace('variable action_start timer',
                        'print "PRE_STATE $(step) $(atoms) $(temp) $(press) $(density) $(etotal) $(v_xshock)"\nvariable action_start timer')
    text = text.replace('run 0 post no', 'run 0 post no\nvariable prep_stop timer')
    text = text.replace('run 500', 'variable balance_stop timer\nrun 500')
    text += ('print "PREP_SECONDS $(v_prep_stop-v_action_start:%.9f)"\n'
             'print "BALANCE_SECONDS $(v_balance_stop-v_prep_stop:%.9f)"\n'
             'print "RUN_SECONDS $(v_action_stop-v_balance_stop:%.9f)"\n')
    return text


def trial(out, job, cps):
    global CHILD
    phase, ck, skin, factor, rep = job
    key = f'{phase}-{ck}-s{skin:g}-w{factor:g}-r{rep}'
    parent = out / 'runs' / key
    parent.mkdir(parents=True, exist_ok=True)
    attempt = parent / f'attempt-{time.time_ns()}'
    attempt.mkdir()
    cp = cps[ck]
    inp = attempt / 'in.lammps'
    inp.write_text(script(cp, skin, factor))
    log = attempt / 'log.lammps'
    command = base.mpi_command(32, inp, log)
    command.insert(1, '--nooversubscribe')
    started = time.time()
    rc = None
    try:
        with (attempt/'stdout.txt').open('w') as stdout, (attempt/'stderr.txt').open('w') as stderr:
            CHILD = subprocess.Popen(command, cwd=attempt, env={**os.environ, 'OMP_NUM_THREADS':'1'},
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                rc = CHILD.wait(timeout=180)
            except subprocess.TimeoutExpired:
                rc = -9
    finally:
        clean_child()
    text = log.read_text() if log.exists() else ''
    parsed = base.parse_trial(log)
    blocks = base.timing_blocks(text)
    parsed['warmup'] = next((b for b in blocks if b['steps']==50), {})
    fields = {}
    for name in ('ACTION_WALL_SECONDS','PREP_SECONDS','BALANCE_SECONDS','RUN_SECONDS'):
        m = re.search(rf'^{name} ([\d.eE+-]+)$', text, re.M)
        fields[name] = float(m[1]) if m else None
    pre = re.search(r'^PRE_STATE (.*)$', text, re.M)
    pre = list(map(float, pre[1].split())) if pre else []
    metric = parsed['measurement']
    builds = re.findall(r'Dangerous builds =\s*(\d+)', text)
    safe = (rc==0 and metric.get('steps')==500 and len(pre)==7
            and all(v is not None and math.isfinite(v) and v>=0 for v in fields.values())
            and builds and all(int(v)==0 for v in builds)
            and all(math.isfinite(v) for v in parsed['thermo_after'].values())
            and not parsed['errors'] and 'weight neigh skipped' not in text
            and parsed['thermo_after']['step']==cp['step']+550
            and pre[1]==138240)
    row = dict(key=key, phase=phase, checkpoint=ck, skin=skin, factor=factor,
               repetition=rep, safe=bool(safe), returncode=rc, timestamp=started,
               process_wall_seconds=time.time()-started, times=fields,
               pre_state=pre, lammps=parsed, command=command, directory=str(attempt))
    (attempt/'result.json').write_text(json.dumps(row, indent=2, allow_nan=True))
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['preflight','main'], required=True)
    a=p.parse_args()
    out=a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock=(out/'lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    cps={c['checkpoint_id']:c for c in json.loads((SOURCE/'checkpoint_states.json').read_text())}
    manifest=dict(version=1, git_commit=base.git_commit(), skins=SKINS, factors=FACTORS,
                  mpi=32, steps=500, warmup=50, diagnostics='CVCF/history excluded in all forks',
                  checkpoints={k:digest(Path(c['restart'])) for k,c in cps.items()},
                  binary_sha256=digest(base.LMP), runner_sha256=digest(Path(__file__)))
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise RuntimeError('Manifest mismatch; use a new output directory')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    data=out/'measurements.jsonl'
    rows=[json.loads(l) for l in data.read_text().splitlines()] if data.exists() else []
    if any(not r['safe'] for r in rows):
        raise RuntimeError('Unsafe record: audit before resuming')
    if a.phase=='main' and sum(r['phase']=='preflight' for r in rows)!=24:
        raise RuntimeError('All 24 endpoint checks must pass first')
    jobs=([(a.phase,ck,s,f,0) for ck in cps for s in (SKINS[0],SKINS[-1]) for f in (FACTORS[0],FACTORS[-1])]
          if a.phase=='preflight' else
          [(a.phase,ck,s,f,r) for r in range(1,6) for ck in cps for s in SKINS for f in FACTORS])
    # Randomize within each repetition block, so temporal drift is distributed.
    rng=random.Random(20260927)
    if a.phase=='main':
        blocks=[jobs[i:i+180] for i in range(0,len(jobs),180)]
        for b in blocks: rng.shuffle(b)
        jobs=[j for b in blocks for j in b]
    else: rng.shuffle(jobs)
    done={r['key'] for r in rows}
    start=time.monotonic()
    with data.open('a') as stream:
        for job in jobs:
            ph,ck,s,f,r=job
            key=f'{ph}-{ck}-s{s:g}-w{f:g}-r{r}'
            if key in done: continue
            if time.monotonic()-start>7*3600: raise RuntimeError('Seven-hour execution budget reached')
            row=trial(out,job,cps)
            stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
            done.add(key)
            print(key, 'safe=',row['safe'], 'seconds=',row['times']['ACTION_WALL_SECONDS'],flush=True)
            if not row['safe']: raise RuntimeError(f'Unsafe or invalid trial: {key}')
    (out/f'{a.phase}_complete.json').write_text(json.dumps(dict(phase=a.phase, trials=len(jobs), completed=time.time())))


if __name__=='__main__':
    try: main()
    finally: clean_child()
