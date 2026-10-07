#!/usr/bin/env python3
"""Cheap axial-size/episode-length probe from an existing shock restart."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import shock_counterfactual as base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restart', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--block-steps', type=int, default=10000)
    ap.add_argument('--parse-only', action='store_true',
                    help='reparse a completed run without launching MPI')
    args = ap.parse_args()
    if args.blocks < 1 or args.block_steps < 1:
        ap.error('blocks and block-steps must be positive')
    restart = args.restart.resolve()
    if not restart.is_file():
        ap.error(f'missing restart: {restart}')
    out = args.output.resolve()
    if args.parse_only:
        prior = json.loads((out / 'result.json').read_text())
        rc, wall, command = prior['return_code'], prior['wall_seconds'], prior['command']
    else:
        out.mkdir(parents=True, exist_ok=False)
        setup = base.fork_input(restart, 0.5, 'none', 500, 50).split('variable action_start timer')[0]
        lines = [setup, 'balance 1.0 shift x 10 1.0 weight neigh 1.5']
        for _ in range(args.blocks):
            lines.append(f'run {args.block_steps}')
        script = out / 'in.probe'
        script.write_text('\n'.join(lines) + '\n')
        command = base.mpi_command(32, script, out / 'log.lammps')
        started = time.monotonic()
        with (out / 'stdout.txt').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
            process = subprocess.Popen(command, cwd=out, stdout=stdout, stderr=stderr,
                                       env={**os.environ, 'OMP_NUM_THREADS': '1'},
                                       start_new_session=True)
            try:
                rc = process.wait(timeout=3600)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        wall = time.monotonic() - started
    log_path = out / 'log.lammps'
    log = log_path.read_text(errors='replace') if log_path.exists() else ''
    # The existing thermo_style includes xshock. Avoid evaluating its
    # collective momentum compute from a separate print command.
    thermo_rows = []
    warmup_endpoint = None
    for line in log.splitlines():
        if warmup_endpoint is None and re.search(r'Loop time of .* for 50 steps', line):
            warmup_endpoint = thermo_rows[-1]['step'] if thermo_rows else None
        fields = line.split()
        if len(fields) != 7 or not re.fullmatch(r'\d+', fields[0]):
            continue
        try:
            step, temp, pe, density, energy, pressure, xshock = map(float, fields)
        except ValueError:
            continue
        thermo_rows.append(dict(step=int(step), temperature=temp, pressure=pressure,
                                density=density, energy=energy, shock_position=xshock))
    by_step = {row['step']: row for row in thermo_rows}
    probes = [by_step[warmup_endpoint + i * args.block_steps]
              for i in range(1, args.blocks + 1)
              if warmup_endpoint + i * args.block_steps in by_step] if warmup_endpoint is not None else []
    bounds = re.search(r'orthogonal box = \(([-\d.eE+]+)\s+[-\d.eE+]+\s+[-\d.eE+]+\) to \(([-\d.eE+]+)', log)
    xlo, xhi = (float(bounds[1]), float(bounds[2])) if bounds else (None, None)
    for probe in probes:
        if xhi is not None:
            probe['front_fraction_x'] = (probe['shock_position'] - xlo) / (xhi - xlo)
    result = dict(return_code=rc, wall_seconds=wall, restart=str(restart),
                  blocks=args.blocks, block_steps=args.block_steps,
                  x_bounds=(xlo, xhi), probes=probes, command=command)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if rc or len(probes) != args.blocks or log.count('Dangerous builds = 0') < args.blocks:
        raise SystemExit('probe failed or incomplete; inspect log and stderr')


if __name__ == '__main__':
    main()
