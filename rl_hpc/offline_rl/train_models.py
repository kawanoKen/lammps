#!/usr/bin/env python3
"""Episode-split training for myopic predictor and sequential fitted Q."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from models import fit_fqi

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=20260928); a=p.parse_args()
    rows=[json.loads(x) for x in (a.dataset/'transitions.jsonl').read_text().splitlines() if x]
    episodes=sorted({x['episode_id'] for x in rows}); rng=random.Random(a.seed); rng.shuffle(episodes)
    split={'train':episodes[:30],'validation':episodes[30:40],'test':episodes[40:]}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/'split.json').write_text(json.dumps(split,indent=2)+'\n')
    train=[x for x in rows if x['episode_id'] in split['train']]
    for name,gamma in [('predictor',0.0),('fqi_myopic',0.0),('fqi_sequential',0.97)]:
        model,loss=fit_fqi(train,gamma)
        model.dump(a.output/f'{name}.json',{'algorithm':'ridge_fqi','gamma':gamma,'iterations':len(loss),'train_episodes':split['train'],'dataset':str(a.dataset),'seed':a.seed,'losses':loss})
    print(json.dumps({'train':len(split['train']),'validation':len(split['validation']),'test':len(split['test']),'transitions':len(train)}))
if __name__=='__main__': main()
