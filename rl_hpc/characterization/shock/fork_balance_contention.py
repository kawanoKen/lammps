#!/usr/bin/env python3
"""Matched idle/contention forks using the existing RL CPU co-runner."""
from __future__ import annotations

import argparse
import collections
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import threading
import time

import psutil

import collect_trajectories as collector
import cpu_jitter
import fork_representative_balance_actions as forks
import shock_counterfactual as base


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def close_simulation(sim, graceful=False):
    """Bounded cleanup, including MPI ranks with separate process groups."""
    process = sim.process
    if graceful and process.poll() is None:
        try:
            process.stdin.write('quit\n')
            process.stdin.flush()
            process.wait(timeout=15)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        for child in descendants:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        _, alive = psutil.wait_procs(descendants, timeout=3)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=3)
        if any(p.status() != psutil.STATUS_ZOMBIE for p in alive):
            raise RuntimeError('MPI cleanup incomplete')
    sim.stderr.close()
    return process.returncode


def key(record, factor, repetition, regime):
    label = 'skip' if factor is None else f'factor-{factor:g}'
    return f"phase-{record['phase']:02d}-{label}-r{repetition}-{regime}"


def monitor_host(output, stop):
    """Stop only this runner if an unrelated user workload becomes active."""
    own = psutil.Process()
    ancestors = {p.pid for p in own.parents()}
    previous = {}
    editor_windows = {}
    index = 0
    while not stop.is_set():
        family = {own.pid, *ancestors, *(p.pid for p in own.children(recursive=True))}
        current = {}
        offenders = []
        editor_activity = []
        for process in psutil.process_iter(['pid', 'name', 'uids', 'cpu_times', 'cmdline']):
            try:
                info = process.info
                cpu = info['cpu_times'].user + info['cpu_times'].system
                current[process.pid] = cpu
                if (process.pid not in family and info['uids'].real >= 1000
                        and info['name'] != 'codex' and process.pid in previous
                        and cpu - previous[process.pid] > .5):
                    item = dict(pid=process.pid, name=info['name'],
                                          cpu_seconds=cpu-previous[process.pid],
                                          create_time=process.create_time(),
                                          affinity=process.cpu_affinity(), command=info['cmdline'])
                    editor = (any('bootstrap-fork' in a for a in info['cmdline'] or [])
                              and '--type=extensionHost' in (info['cmdline'] or []))
                    if editor:
                        editor_activity.append(item)
                    else:
                        offenders.append(item)
                editor = (any('bootstrap-fork' in a for a in info['cmdline'] or [])
                          and '--type=extensionHost' in (info['cmdline'] or []))
                if editor and process.pid in previous:
                    window = editor_windows.setdefault(process.pid, collections.deque(maxlen=5))
                    window.append(cpu-previous[process.pid])
                    if sum(window) > 2.5:
                        offenders.append(dict(pid=process.pid, name=info['name'],
                                              editor_cpu_seconds_last_5_checks=sum(window),
                                              command=info['cmdline']))
            except (psutil.Error, TypeError):
                continue
        gpu = None
        if index % 5 == 0:
            check = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            if check.returncode != 0:
                offenders.append(dict(error='GPU monitor failed', stderr=check.stderr))
            gpu = check.stdout.strip()
            if gpu:
                offenders.append(dict(gpu_compute_pids=gpu))
        with (output/'host_monitor.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), cpu_offenders=offenders,
                                         editor_activity=editor_activity,
                                         gpu_compute_pids=gpu, excluded_family_count=len(family)))+'\n')
        if offenders:
            (output/'external_load_stop.json').write_text(json.dumps(offenders, indent=2)+'\n')
            os.kill(os.getpid(), signal.SIGINT)
            return
        previous = current
        index += 1
        stop.wait(2)


def run_trial(output, record, params, factor, repetition, regime, cores):
    identifier = key(record, factor, repetition, regime)
    directory = output / 'runs' / identifier
    directory.mkdir(parents=True)
    setup, measured = forks.input_text(record, params, factor).split('variable begin timer', 1)
    measured = 'variable begin timer' + measured
    (directory / 'setup.lammps').write_text(setup)
    (directory / 'measurement.lammps').write_text(measured)
    contender = cpu_jitter.CpuJitter(cores) if regime == 'contention' else None
    sim = None
    started = time.time()
    row = dict(key=identifier, phase=record['phase'], checkpoint_step=record['step'],
               checkpoint_sha256=record['sha256'], repetition=repetition, regime=regime,
               action=dict(balance=factor is not None, factor=factor, skin=0.5),
               safe=False, returncode=None, errors=[], timestamp=started)
    try:
        if contender:
            contender.start()  # Inactive during restart and common partition reconstruction.
        sim = collector.Simulation(directory)
        before = sim.execute(setup)
        (directory / 'setup.stdout.txt').write_text(before)
        pre = list(map(float, re.findall(r'^PRE_STATE (.*)$', before, re.M)[-1].split()))
        if (len(pre) != 7 or not finite(pre) or pre[0] != record['step']
                or pre[1] != 491520 or pre[2] <= 0 or pre[4] <= 0
                or re.search(r'^ERROR', before, re.M)):
            raise RuntimeError('invalid reconstructed pre-state')
        ranks = [p for p in psutil.Process(sim.process.pid).children(recursive=True)
                 if p.name() == 'lmp']
        bindings = [dict(pid=p.pid, cpus=p.cpu_affinity()) for p in ranks]
        physical = [{int(Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text())
                     for c in b['cpus']} for b in bindings]
        if len(ranks) != 32 or any(len(c) != 1 for c in physical) or len(set.union(*physical)) != 32:
            raise RuntimeError('MPI32 physical core placement invalid')
        occupied = set().union(*(set(b['cpus']) for b in bindings))
        if not set(cores).issubset(occupied):
            raise RuntimeError('contention cores do not overlap measured MPI placement')
        row.update(pre_thermo_reconstructed=pre, mpi_bindings=bindings,
                   reconstructed_mesh_sha256=forks.digest(directory / 'reconstructed.mesh'))
        if contender:
            contender.set_active(True)
            time.sleep(.15)  # Existing RL decision-boundary settling interval; outside timer.
        worker_before = contender.cpu_seconds() if contender else 0.0
        text = sim.execute(measured)
        worker_after = contender.cpu_seconds() if contender else 0.0
        if contender:
            contender.set_active(False)
        (directory / 'measurement.stdout.txt').write_text(text)
        state = collector.observation(text, 0.5, factor, expected_atoms=491520)
        times = list(map(float, re.findall(r'^TIMES (.*)$', text, re.M)[-1].split()))
        rc = close_simulation(sim, graceful=True)
        row.update(returncode=rc, runtime_seconds=times[0], reward=-times[0],
                   preparation_seconds=times[1], balance_seconds=times[2], run_seconds=times[3],
                   next_state=dict(thermo=base.thermo(text), lammps=state['previous_segment'],
                                   observation=state),
                   worker_cpu_seconds=worker_after-worker_before,
                   cpu_contention_active=regime == 'contention')
        numeric_error = re.search(r'(?i)(?<![a-z_])(?:nan|[+-]?inf(?:inity)?)(?![a-z_])', before + text)
        errors = re.findall(r'^ERROR.*$', before + text, re.M)
        row['errors'] = errors
        if (rc != 0 or len(times) != 4 or not finite(times) or min(times) < 0
                or abs(times[0]-sum(times[1:])) > 1e-6 or state['step'] != record['step']+500
                or state['previous_segment']['mpi_ranks'] != 32
                or not finite(row['next_state']) or numeric_error or errors
                or (contender and row['worker_cpu_seconds'] <= 0)):
            raise RuntimeError('invalid or unsafe measurement')
        row['safe'] = True
    except Exception as exc:
        row['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        if sim:
            rc = close_simulation(sim)
            row['returncode'] = rc
        if contender:
            contender.close()
            row['workers_reaped'] = all(not p.is_alive() for p in contender.children)
        row['process_wall_seconds'] = time.time()-started
        (directory / 'result.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'main'), required=True)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--cpu-cores', default='0-7')
    parser.add_argument('--monitor-host', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cores = cpu_jitter.parse_cores(args.cpu_cores)
    source = args.checkpoints.resolve() / 'manifest.json'
    manifest = json.loads(source.read_text())
    records = manifest['records']
    if len(records) != 8 or args.repetitions != 5:
        parser.error('requires eight checkpoints and five repetitions')
    for record in records:
        if forks.digest(Path(record['restart'])) != record['sha256']:
            parser.error('checkpoint SHA256 mismatch')
    params = Path(manifest['initial_restart']['path']).parent / 'shockparams.mod'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = [records[0], records[-1]] if args.phase == 'smoke' else records
    factors = (None, .5, 1.5) if args.phase == 'smoke' else forks.FACTORS
    repetitions = 1 if args.phase == 'smoke' else args.repetitions
    rng = random.Random(args.seed)
    jobs = []
    for repetition in range(repetitions):
        block = [(r, f, repetition, regime) for r in selected for f in factors
                 for regime in ('idle', 'contention')]
        rng.shuffle(block)
        jobs.extend(block)
    protocol = dict(version='shock-matched-contention-v1', git_commit=base.git_commit(),
                    mpi_ranks=32, omp_threads=1, steps=500, skin=.5, factors=forks.FACTORS,
                    repetitions=repetitions, seed=args.seed, contention_cores=cores,
                    source_manifest_sha256=forks.digest(source), binary_sha256=forks.digest(base.LMP),
                    runner_sha256=forks.digest(Path(__file__)),
                    fork_source_sha256=forks.digest(Path(forks.__file__)),
                    co_runner_sha256=forks.digest(Path(cpu_jitter.__file__)),
                    schedule=[key(*job) for job in jobs],
                    contention='continuous active during measured action+500 steps; RL CpuJitter',
                    host_monitor=args.monitor_host,
                    host_monitor_policy='non-editor >0.5 CPU seconds/check; VSCode extensionHost >2.5 CPU seconds/5 checks',
                    pre_action='weight neigh 1.0; contender inactive; outside reward timer')
    target = output/'manifest.json'
    normalized = json.loads(json.dumps(protocol))
    if target.exists() and json.loads(target.read_text()) != normalized:
        old = json.loads(target.read_text())
        allowed = {'runner_sha256', 'host_monitor_policy'}
        differences = {k for k in set(old) | set(normalized) if old.get(k) != normalized.get(k)}
        if not args.resume or args.phase != 'main' or not differences.issubset(allowed):
            raise RuntimeError('manifest mismatch')
        stamp = time.time_ns()
        (output/f'manifest.before-resume-{stamp}.json').write_text(target.read_text())
        (output/f'resume_audit-{stamp}.json').write_text(json.dumps(dict(
            time=time.time(),changed_fields=sorted(differences),old_runner_sha256=old['runner_sha256'],
            new_runner_sha256=normalized['runner_sha256'],measurement_conditions_unchanged=True),indent=2)+'\n')
    target.write_text(json.dumps(protocol, indent=2)+'\n')
    data = output/'measurements.jsonl'
    rows = [json.loads(s) for s in data.read_text().splitlines()] if data.exists() else []
    if any(not r['safe'] for r in rows):
        raise RuntimeError('unsafe previous trial; stop')
    done = {r['key'] for r in rows}
    if args.resume:
        if not (output/'external_load_stop.json').exists():
            raise RuntimeError('resume requires preserved external-load stop evidence')
        for directory in (output/'runs').iterdir():
            if directory.name not in done:
                result = directory/'result.json'
                if not result.exists():
                    raise RuntimeError('unclassified incomplete trial; audit required')
                row = json.loads(result.read_text())
                if row.get('errors') or row.get('runtime_seconds') is not None:
                    raise RuntimeError('failed measurement cannot be automatically resumed')
                log = (directory/'log.lammps').read_text(errors='replace')
                if (re.search(r'^ERROR', log, re.M)
                        or re.search(r'(?i)(?<![a-z_])(?:nan|[+-]?inf(?:inity)?)(?![a-z_])', log)
                        or any(int(v)>0 for v in re.findall(r'Dangerous builds =\s*(\d+)', log))):
                    raise RuntimeError('unsafe interrupted log; no resume')
                archive = output/'interrupted_runs'
                archive.mkdir(exist_ok=True)
                directory.rename(archive/f'{directory.name}-{time.time_ns()}')
        for name in ('external_load_stop.json','main_complete.json'):
            path=output/name
            if path.exists():
                path.rename(output/f'{name}.before-resume-{time.time_ns()}')
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_host, args=(output, stop), daemon=True) if args.monitor_host else None
    if monitor:
        monitor.start()
    try:
        with data.open('a') as stream:
            for job in jobs:
                if key(*job) in done:
                    continue
                if monitor and not monitor.is_alive():
                    raise RuntimeError('host monitor stopped unexpectedly')
                row = run_trial(output, job[0], params, *job[1:], cores)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
                print(row['key'], 'safe', row['safe'], 'seconds', row.get('runtime_seconds'), flush=True)
                if not row['safe']:
                    raise RuntimeError(f"unsafe trial: {row['key']}: {row['errors']}")
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=12)
    for record in records:
        if forks.digest(Path(record['restart'])) != record['sha256']:
            raise RuntimeError('checkpoint changed during experiment')
    (output/f'{args.phase}_complete.json').write_text(json.dumps(dict(trials=len(jobs), time=time.time()))+'\n')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
