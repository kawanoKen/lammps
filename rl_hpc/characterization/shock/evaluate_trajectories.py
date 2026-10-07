#!/usr/bin/env python3
"""Paired live evaluation of frozen shock policies; no online learning."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import collect_trajectories as collect
import numeric_sweep as sweep
import shock_counterfactual as base
import train_trajectories as train

POLICIES=('fixed','rule','predictor','fqi','fixed_05')
FIXED=(.35,1.5)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--pairs',type=int,default=4);ap.add_argument('--decisions',type=int,default=25)
    ap.add_argument('--restart',type=Path,default=sweep.SOURCE/'checkpoint-S1.restart')
    ap.add_argument('--expected-atoms',type=int,default=138240)
    ap.add_argument('--policies',nargs='+',choices=POLICIES,default=['fixed','rule','predictor','fqi'])
    args=ap.parse_args();assert args.decisions>0 and args.pairs>0
    if args.decisions!=25 and any(p in ('predictor','fqi') for p in args.policies):
        ap.error('Learned policies are calibrated for exactly 25 decisions')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    model_dir=base.ROOT/'rl_hpc/characterization/data/shock_training_v1'
    models={}
    for name,label in [('predictor','predictor_myopic'),('fqi','fqi_sequential')]:
        if name not in args.policies:continue
        path=model_dir/f'{label}_seed22.npz'; d=np.load(path)
        models[name]=(d['coef'],d['mean'],d['scale'])
    manifest={'policies':args.policies,'fixed':FIXED,'pairs':args.pairs,'decisions':args.decisions,
              'mpi':32,'steps_per_decision':500,'restart':str(args.restart.resolve()),'expected_atoms':args.expected_atoms,
              'model_hashes':{p:hashlib.sha256((model_dir/f'{q}_seed22.npz').read_bytes()).hexdigest()
                              for p,q in [('predictor','predictor_myopic'),('fqi','fqi_sequential')] if p in args.policies},
              'rule':'imbalance>1.2 -> (.5,1.5), else (.35,.8)',
              'criterion':'paired total action+run timer, no startup/cooldown; separate process per policy',
              'scope':'one physical restart, varied execution ordering only'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def choose(policy,s,t):
        if policy=='fixed':return FIXED
        if policy=='fixed_05':return (.5,1.5)
        if policy=='rule':return collect.rule(s)
        coef,mean,scale=models[policy]
        z=(np.asarray(train.state(s,t))-mean)/scale
        q=train.candidates(z[None,:])[0]@coef
        assert np.isfinite(q).all()
        return train.ACTIONS[int(q.argmax())]
    def run(pair,policy):
        directory=args.output/f'pair-{pair:02d}-{policy}';directory.mkdir()
        sim=collect.Simulation(directory);start=time.time();complete=False
        try:
            restart=args.restart.resolve();initial=FIXED
            setup=base.fork_input(restart,initial[0],'none',500,50).split('variable action_start timer')[0]
            setup+=f'\nneighbor {initial[0]} bin\nrun 0 post no\nbalance 1.0 shift x 10 1.0 weight neigh {initial[1]} out initial.mesh\nrun 50\n'+collect.OBS
            s=collect.observation(sim.execute(setup),*initial,expected_atoms=args.expected_atoms)
            s['partition_mesh']=(directory/'initial.mesh').read_text()
            rows=[]
            with (directory/'transitions.jsonl').open('w') as stream:
                for t in range(args.decisions):
                    skin,factor=choose(policy,s,t)
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
{collect.OBS}
'''
                    wall=time.monotonic();output=sim.execute(commands);external=time.monotonic()-wall
                    (directory/f'segment-{t:03d}.txt').write_text(output)
                    ns=collect.observation(output,skin,factor,expected_atoms=args.expected_atoms)
                    ns['partition_mesh']=(directory/f'partition-{t:03d}.mesh').read_text()
                    assert ns['step']==s['step']+500
                    ts=[float(x) for x in re.findall(r'^TIMES (.*)$',output,re.M)[-1].split()]
                    assert len(ts)==4 and all(math.isfinite(v) and v>=0 for v in ts)
                    assert abs(ts[0]-sum(ts[1:]))<1e-6
                    row={'pair':pair,'policy':policy,'decision':t,'state_step':s['step'],'action':(skin,factor),
                         'runtime':ts[0],'prep':ts[1],'balance':ts[2],'run':ts[3],
                         'external_seconds':external,'next_step':ns['step'],'temperature':ns['temperature'],
                         'pressure':ns['pressure'],'energy':ns['total_energy_per_atom'],
                         'shock_position':ns['shock_position'],'dangerous_builds':ns['previous_segment']['dangerous_builds']}
                    stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row);s=ns
            rc=sim.close(graceful=True);assert rc==0
            complete=True
            result={'pair':pair,'policy':policy,'total_seconds':sum(r['runtime'] for r in rows),
                    'external_total_seconds':time.time()-start,'end_step':s['step'],
                    'dangerous_builds':sum(r['dangerous_builds'] for r in rows),'exit_code':rc}
            (directory/'result.json').write_text(json.dumps(result,indent=2))
            return result
        finally:
            sim.close();(directory/'status.json').write_text(json.dumps({'complete':complete,'time':time.time()}))
    results=[]
    for pair in range(args.pairs):
        order=list(args.policies);random.Random(9321+pair).shuffle(order)
        for policy in order:
            result=run(pair,policy);results.append(result)
            (args.output/'results.json').write_text(json.dumps(results,indent=2))
            print(pair,policy,result['total_seconds'],flush=True)


if __name__=='__main__':main()
