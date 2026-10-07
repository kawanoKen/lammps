#!/usr/bin/env python3
import argparse,collections,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
rows=[]
for f in sorted(a.directory.glob('episode-*.jsonl')):rows += [json.loads(x) for x in f.read_text().splitlines() if x]
(a.directory/'transitions.partial.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
def clipped(row):
    requested, applied = row['requested_action'], row['applied_action']
    return abs(float(requested[0])-float(applied[0])) > 1e-10 or int(requested[1]) != int(applied[1])
out={'episodes':len(set(x['episode_id'] for x in rows)),'transitions':len(rows),'regimes':dict(collections.Counter(x['contention_ground_truth'] for x in rows)),'every_delta':dict(collections.Counter(x['requested_action'][1] for x in rows)),'clipping':sum(clipped(x) for x in rows),'unsafe':sum(x['unsafe'] for x in rows),'runtime_mean_s':statistics.mean(x['segment_runtime_seconds'] for x in rows),'runtime_std_s':statistics.stdev(x['segment_runtime_seconds'] for x in rows),'leak':any('contention_ground_truth' in x['state_before'] for x in rows)}
(a.directory/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
