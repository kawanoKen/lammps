#!/usr/bin/env python3
"""Audit and summarize paired hybrid-balance live evaluation."""
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
               if (root / 'results.json').is_file() else [])
    valid = []
    action_stats = {}
    for result in results:
        directory = root / result['id']
        status = json.loads((directory / 'status.json').read_text())
        rows = [json.loads(line) for line in
                (directory / 'transitions.jsonl').read_text().splitlines()]
        assert status['status'] == 'complete'
        assert len(rows) == result['decisions'] == manifest['decisions']
        assert all(row['decision'] == index and
                   row['step_end'] - row['step_start'] == 500 and
                   row['dangerous_builds'] == 0 and
                   math.isfinite(row['action_wall_seconds'])
                   for index, row in enumerate(rows))
        assert abs(sum(row['action_wall_seconds'] for row in rows) -
                   result['total_seconds']) < 1e-5
        schedule_key = f'{result["physical_seed"]}-{result["regime"]}-r{result["repeat"]}'
        assert [row['cpu_contention_active'] for row in rows] == \
            manifest['schedule'][schedule_key]['active_by_decision']
        valid.append(result)
        factors = [row['action']['factor'] for row in rows
                   if row['action']['balance']]
        action_stats[result['id']] = {
            'balance_count': len(factors),
            'skip_count': len(rows) - len(factors),
            'mean_factor_when_balancing': statistics.mean(factors) if factors else None,
            'min_factor': min(factors) if factors else None,
            'max_factor': max(factors) if factors else None,
        }
    expected = (len(manifest['heldout_restarts']) * len(manifest['regimes']) *
                len(manifest['policies']) * manifest['repeats'])
    output = {'complete': len(valid) == expected, 'completed_runs': len(valid),
              'expected_runs': expected, 'regimes': {}, 'paired_comparisons': [],
              'action_stats': action_stats}
    reference = 'fixed_15'
    for regime in manifest['regimes']:
        subset = [row for row in valid if row['regime'] == regime]
        indexed = {(row['physical_seed'], row['repeat'], row['policy']):
                   row['total_seconds'] for row in subset}
        output['regimes'][regime] = {}
        for policy in manifest['policies']:
            values = [row['total_seconds'] for row in subset if row['policy'] == policy]
            balance_counts = [row['balance_count'] for row in subset
                              if row['policy'] == policy]
            output['regimes'][regime][policy] = {
                'n': len(values),
                'mean_seconds': statistics.mean(values) if values else None,
                'std_seconds': statistics.stdev(values) if len(values) > 1 else None,
                'mean_balance_count': statistics.mean(balance_counts)
                if balance_counts else None,
            }
            if policy == reference:
                continue
            differences = []
            for spec in manifest['heldout_restarts']:
                for repeat in range(manifest['repeats']):
                    key = (spec['physical_seed'], repeat)
                    if ((*key, policy) in indexed and (*key, reference) in indexed):
                        differences.append(indexed[(*key, reference)] -
                                           indexed[(*key, policy)])
            output['paired_comparisons'].append({
                'regime': regime, 'policy': policy, 'reference': reference,
                'n': len(differences), 'paired_savings_seconds': differences,
                'mean_savings_seconds': statistics.mean(differences)
                if differences else None,
                'std_savings_seconds': statistics.stdev(differences)
                if len(differences) > 1 else None,
            })
    (root / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
