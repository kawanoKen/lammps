#!/usr/bin/env python3
"""Audit persistent shock trajectory datasets without training a model."""
import argparse
import collections
import json
import math
from pathlib import Path

import collect_trajectories as collect


def main():
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);a=ap.parse_args()
    root=a.dataset.resolve();manifest=json.loads((root/'manifest.json').read_text())
    decisions=manifest['decisions'];bins=math.ceil(decisions/5)
    coverage=[collections.Counter() for _ in range(bins)]
    results=[];all_rows=[]
    for directory in sorted(root.glob('episode-*')):
        rows=[json.loads(l) for l in (directory/'transitions.jsonl').read_text().splitlines()]
        status=json.loads((directory/'status.json').read_text())
        assert status['status']=='complete' and len(rows)==decisions
        assert not (directory/'failure.json').exists()
        for i,r in enumerate(rows):
            s,n=r['state'],r['next_state']
            assert r['decision']==i and n['step']-s['step']==500
            assert i==0 or rows[i-1]['next_state']==s
            assert s['atoms']==manifest['expected_atoms']==n['atoms']
            assert r['safe'] and r['reward']==-r['runtime'] and r['runtime']>0
            assert n['previous_segment']['dangerous_builds']==0
            assert all(math.isfinite(n[k]) for k in ('temperature','pressure','density','total_energy_per_atom','shock_position'))
            assert n['temperature']>0 and n['density']>0
            assert abs(r['runtime']-sum(r[k] for k in ('preparation_seconds','balance_seconds','run_seconds')))<1e-6
            action=(r['action']['skin'],r['action']['factor'])
            assert action in collect.ACTIONS
            expected=r['epsilon']/len(collect.ACTIONS)+(1-r['epsilon'] if list(action)==r['rule_action'] else 0)
            assert abs(r['behavior_probability']-expected)<1e-12
            coverage[i//5][action]+=1
        results.append({'episode':directory.name,'start_step':rows[0]['state']['step'],
                        'end_step':rows[-1]['next_state']['step'],
                        'seconds':sum(r['runtime'] for r in rows),'exploration_count':sum(r['exploratory'] for r in rows)})
        all_rows+=rows
    assert len(results)==len(json.loads((root/'summary.json').read_text())['episodes'])
    report={'episodes':len(results),'transitions':len(all_rows),'expected_atoms':manifest['expected_atoms'],
            'decisions':decisions,'all_checks_passed':True,'episodes_detail':results,
            'unique_actions':len({(r['action']['skin'],r['action']['factor']) for r in all_rows}),
            'visits_per_5_decisions':[len(c) for c in coverage],
            'missing_phase_action_cells':sum(len(collect.ACTIONS)-len(c) for c in coverage),
            'exploration_count':sum(r['exploratory'] for r in all_rows),
            'temperature_range':[min(r['next_state']['temperature'] for r in all_rows),max(r['next_state']['temperature'] for r in all_rows)],
            'energy_per_atom_range':[min(r['next_state']['total_energy_per_atom'] for r in all_rows),max(r['next_state']['total_energy_per_atom'] for r in all_rows)],
            'front_range':[min(r['next_state']['shock_position'] for r in all_rows),max(r['next_state']['shock_position'] for r in all_rows)]}
    (root/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='episodes_detail'},indent=2))


if __name__=='__main__':main()
