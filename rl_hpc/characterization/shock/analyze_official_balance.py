#!/usr/bin/env python3
"""Audit official fix-balance comparison and summarize paired runtimes."""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('dataset',type=Path)
    root=ap.parse_args().dataset.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    results=json.loads((root/'results.json').read_text()) if (root/'results.json').exists() else []
    valid=[]
    for result in results:
        directory=root/result['id']
        status=json.loads((directory/'status.json').read_text())
        rows=[json.loads(line) for line in (directory/'transitions.jsonl').read_text().splitlines()]
        assert status['status']=='complete'
        assert len(rows)==result['decisions']==manifest['decisions']
        assert all(row['decision']==i and row['step_end']-row['step_start']==500
                   and row['dangerous_builds']==0 and math.isfinite(row['action_wall_seconds'])
                   for i,row in enumerate(rows))
        assert abs(sum(row['action_wall_seconds'] for row in rows)-result['total_seconds'])<1e-5
        key=f'{result["physical_seed"]}-{result["regime"]}-r{result["repeat"]}'
        assert [row['cpu_contention_active'] for row in rows]==manifest['schedule'][key]['active_by_decision']
        valid.append(result)
    expected=len(manifest['heldout_restarts'])*len(manifest['regimes'])*len(manifest['policies'])*manifest['repeats']
    output={'complete':len(valid)==expected,'completed_runs':len(valid),'expected_runs':expected,
            'regimes':{},'paired_vs_best':[]}
    for regime in manifest['regimes']:
        subset=[row for row in valid if row['regime']==regime]
        summary={}
        for policy in manifest['policies']:
            vals=[row['total_seconds'] for row in subset if row['policy']==policy]
            summary[policy]={'n':len(vals),'mean_seconds':statistics.mean(vals) if vals else None,
                             'std_seconds':statistics.stdev(vals) if len(vals)>1 else None}
        output['regimes'][regime]=summary
        complete_policies=[p for p,v in summary.items() if v['n']==len(manifest['heldout_restarts'])*manifest['repeats']]
        if complete_policies:
            best=min(complete_policies,key=lambda p:summary[p]['mean_seconds'])
            indexed={(r['physical_seed'],r['repeat'],r['policy']):r['total_seconds'] for r in subset}
            for policy in complete_policies:
                if policy==best: continue
                dif=[]
                for spec in manifest['heldout_restarts']:
                    for repeat in range(manifest['repeats']):
                        dif.append(indexed[(spec['physical_seed'],repeat,policy)]-
                                   indexed[(spec['physical_seed'],repeat,best)])
                output['paired_vs_best'].append({'regime':regime,'policy':policy,'best':best,
                    'paired_penalty_seconds':dif,'mean_penalty_seconds':statistics.mean(dif),
                    'std_penalty_seconds':statistics.stdev(dif) if len(dif)>1 else None})
    (root/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
