#!/usr/bin/env python3
"""Audit paired frozen-policy shock evaluation and report paired differences."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    results = (json.loads((root / 'results.json').read_text())
               if (root / 'results.json').exists() else [])
    validated = []
    for result in results:
        directory = root / result['id']
        status = json.loads((directory / 'status.json').read_text())
        rows = [json.loads(line) for line in
                (directory / 'transitions.jsonl').read_text().splitlines()]
        assert status['status'] == 'complete'
        assert len(rows) == result['decisions'] == manifest['decisions']
        assert all(row['decision'] == i and row['step_end'] - row['step_start'] == 500
                   and row['dangerous_builds'] == 0 and math.isfinite(row['action_wall_seconds'])
                   for i, row in enumerate(rows))
        assert abs(sum(row['action_wall_seconds'] for row in rows) -
                   result['total_seconds']) < 1e-5
        schedule_key = f'{result["physical_seed"]}-{result["regime"]}-r{result["repeat"]}'
        assert [row['cpu_contention_active'] for row in rows] == \
            manifest['schedule'][schedule_key]['active_by_decision']
        validated.append(result)
    expected = len(manifest['heldout_restarts']) * len(manifest['regimes']) * \
        len(manifest['policies']) * manifest['repeats']
    output = {'complete': len(validated) == expected,
              'completed_runs': len(validated), 'expected_runs': expected,
              'paired_comparisons': []}
    for regime in manifest['regimes']:
        subset = [row for row in validated if row['regime'] == regime]
        by_policy = {policy: [row['total_seconds'] for row in subset
                              if row['policy'] == policy] for policy in manifest['policies']}
        output.setdefault('regimes', {})[regime] = {
            policy: {'n': len(values), 'mean_seconds': statistics.mean(values) if values else None,
                     'std_seconds': statistics.stdev(values) if len(values) > 1 else None}
            for policy, values in by_policy.items()}
        indexed = {(row['physical_seed'], row['repeat'], row['policy']): row['total_seconds']
                   for row in subset}
        for policy in manifest['policies']:
            if policy == 'fixed_05_15':
                continue
            differences = []
            for spec in manifest['heldout_restarts']:
                for repeat in range(manifest['repeats']):
                    key = (spec['physical_seed'], repeat)
                    if ((*key, policy) in indexed and (*key, 'fixed_05_15') in indexed):
                        differences.append(indexed[(*key, 'fixed_05_15')] -
                                           indexed[(*key, policy)])
            output['paired_comparisons'].append({
                'regime': regime, 'policy': policy, 'reference': 'fixed_05_15',
                'n': len(differences), 'paired_savings_seconds': differences,
                'mean_savings_seconds': statistics.mean(differences) if differences else None,
                'std_savings_seconds': statistics.stdev(differences)
                if len(differences) > 1 else None})
    (root / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
