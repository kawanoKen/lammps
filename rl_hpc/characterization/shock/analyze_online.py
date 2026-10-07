#!/usr/bin/env python3
"""Audit online shock transitions and export reward/action time series."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', type=Path)
    args = p.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    traces, episodes, all_rows, updates = [], [], [], []
    for directory in sorted(root.glob('episode-*')):
        source = directory / 'transitions.jsonl'
        if not source.exists():
            continue
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if not rows:
            continue
        schedule = None
        if manifest.get('jitter') == 'cpu_piecewise':
            schedule = json.loads((directory / 'contention_schedule.json').read_text())
            assert len(schedule['active_by_decision']) == manifest['decisions']
        update_path = directory / 'model_updates.jsonl'
        if update_path.exists():
            updates.extend(dict(episode=rows[0]['episode'], **json.loads(line))
                           for line in update_path.read_text().splitlines())
        status_path = directory / 'status.json'
        complete = (status_path.exists() and
                    json.loads(status_path.read_text())['status'] == 'complete' and
                    len(rows) == manifest['decisions'])
        cumulative = 0.0
        for i, row in enumerate(rows):
            assert row['decision'] == i and row['step_end'] - row['step_start'] == 500
            if schedule is not None:
                assert row['cpu_contention_active'] == schedule['active_by_decision'][i]
                assert row['cpu_contention_cores'] == schedule['cores']
                assert row['cpu_contention_worker_seconds'] >= 0
                assert len(row['cpu_utilization_target_cores_before']) == len(schedule['cores'])
            if i:
                assert row['step_start'] == rows[i-1]['step_end']
                assert row['state'] == rows[i-1]['next_state']
            assert row['dangerous_builds'] == 0 and row['safe']
            assert row['reward'] < 0 and math.isfinite(row['reward'])
            assert len(row['q_values_before']) == len(manifest['actions'])
            assert abs(-row['reward'] - sum(row[k] for k in
                       ('prepare_seconds', 'balance_seconds', 'run_seconds'))) < 1e-6
            cumulative += row['reward']
            assert abs(cumulative - row['cumulative_reward']) < 1e-5
            action = row['action']
            assert action == manifest['actions'][row['action_index']]
            traces.append(dict(episode=row['episode'], decision=i,
                               phase=row['phase'], step_start=row['step_start'],
                               step_end=row['step_end'], reward_seconds=row['reward'],
                               cumulative_reward_seconds=row['cumulative_reward'],
                               rolling_10_reward_seconds=row.get(
                                   'rolling_10_reward', statistics.mean(
                                       r['reward'] for r in rows[max(0, i-9):i+1])),
                               segment_wall_seconds=row['action_wall_seconds'],
                               prepare_seconds=row['prepare_seconds'],
                               balance_seconds=row['balance_seconds'],
                               run_seconds=row['run_seconds'],
                               shock_position=row['next_state']['shock_position'],
                               atom_imbalance=row['state']['atom_imbalance'],
                               temperature=row['next_state']['temperature'],
                               pressure=row['next_state']['pressure'],
                               pair_seconds=row['next_state']['previous_segment']['timing_avg_seconds'].get('pair', 0),
                               neigh_seconds=row['next_state']['previous_segment']['timing_avg_seconds'].get('neigh', 0),
                               comm_seconds=row['next_state']['previous_segment']['timing_avg_seconds'].get('comm', 0),
                               neighbor_builds=row['neighbor_builds'],
                               skin=action['skin'], balance=action['balance'],
                               neighbor_factor=action['neighbor_weight_factor'],
                               action_source=row['action_source'], epsilon=row['epsilon'],
                               model_updates_before=row['model_updates_before']))
        episodes.append(dict(episode=rows[0]['episode'], phase=rows[0]['phase'],
                             complete=complete,
                             decisions=len(rows), total_reward_seconds=cumulative,
                             total_action_wall_seconds=-cumulative,
                             mean_reward_seconds=statistics.mean(r['reward'] for r in rows),
                             std_reward_seconds=statistics.stdev(r['reward'] for r in rows)
                             if len(rows)>1 else 0,
                             explored=sum(r['action_source'] != 'greedy' for r in rows),
                             ending_shock_position=rows[-1]['next_state']['shock_position']))
        all_rows.extend(rows)
    if not traces:
        raise SystemExit('no completed transitions')
    write_csv(root / 'reward_trace.csv', traces, list(traces[0]))
    write_csv(root / 'episode_reward_summary.csv', episodes, list(episodes[0]))
    if updates:
        write_csv(root / 'model_update_trace.csv', updates, list(updates[0]))
    by_decision = collections.defaultdict(list)
    for row in traces:
        by_decision[(row['phase'], row['decision'])].append(row['reward_seconds'])
    curve = [dict(phase=phase, decision=decision, count=len(values),
                  mean_reward_seconds=statistics.mean(values),
                  std_reward_seconds=statistics.stdev(values) if len(values)>1 else 0)
             for (phase, decision), values in sorted(by_decision.items())]
    write_csv(root / 'reward_by_decision.csv', curve, list(curve[0]))
    counts = collections.Counter((r['phase'], r['action_index']) for r in all_rows)
    contention = collections.defaultdict(list)
    for row in all_rows:
        if 'cpu_contention_active' in row:
            contention[bool(row['cpu_contention_active'])].append(row)
    contention_summary = {
        str(active).lower(): {
            'transitions': len(values),
            'mean_reward_seconds': statistics.mean(r['reward'] for r in values),
            'mean_worker_cpu_seconds': statistics.mean(
                r.get('cpu_contention_worker_seconds', 0) for r in values),
        } for active, values in contention.items()
    }
    report = dict(complete_episodes=sum(e['complete'] for e in episodes),
                  partial_episodes=sum(not e['complete'] for e in episodes),
                  logged_transitions=len(all_rows),
                  all_checks_passed=True, exploration_episodes=sum(
                      e['phase']=='exploration' for e in episodes),
                  learning_episodes=sum(e['phase']=='online_learning' for e in episodes),
                  reward_min=min(r['reward'] for r in all_rows),
                  reward_max=max(r['reward'] for r in all_rows),
                  model_updates_logged=len(updates),
                  latest_model_td_mse=updates[-1]['train_td_mse'] if updates else None,
                  contention_summary=contention_summary,
                  action_counts={f'{phase}:{index}':count for (phase,index),count in counts.items()},
                  warnings=['Raw reward changes with shock phase; learning improvement requires paired live fixed-policy evaluation.'])
    (root / 'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
