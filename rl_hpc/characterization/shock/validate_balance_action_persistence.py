#!/usr/bin/env python3
"""Validate strong action reversals and measure partition-effect persistence."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

import shock_counterfactual as base
from fork_representative_balance_actions import clean_child, digest
import fork_representative_balance_actions as forks


ACTIONS = {
    1: (None, 1.0),
    3: (1.5, None),
    4: (1.25, 1.0),
    7: (0.75, 0.5),
    8: (0.75, 0.5),
}


def input_text(record: dict, params: Path, factor: float | None,
               continuation_segments: int) -> str:
    action = ("" if factor is None else
              f"balance 1.0 shift x 10 1.0 weight neigh {factor:g} out action.mesh")
    lines = [f"include {params}", f"read_restart {record['restart']}",
             "fix mynve all nve", "fix xwalls all wall/reflect xlo EDGE xhi EDGE",
             "compute mymom all momentum",
             "variable xshock equal (1.0+c_mymom[1]/(atoms*${mass})/${up})*lx-${up}*time",
             "timer normal", "neighbor 0.5 bin",
             "neigh_modify every 1 delay 0 check yes", "thermo 100",
             "thermo_style custom step atoms temp pe density etotal press v_xshock",
             "run 0 post no",
             "balance 1.0 shift x 10 1.0 weight neigh 1.0 out reconstructed.mesh",
             "run 0 post no", "variable b0 timer", "neighbor 0.5 bin",
             "neigh_modify every 1 delay 0 check yes", "run 0 post no",
             "variable p0 timer", action, "variable a0 timer", "run 500",
             "variable e0 timer",
             'print "SEGMENT 0 $(v_e0-v_b0:%.9f) $(v_p0-v_b0:%.9f) '
             '$(v_a0-v_p0:%.9f) $(v_e0-v_a0:%.9f)"',
             'print "OBS_SEG 0 $(step) $(atoms) $(temp) $(press) $(density) $(etotal) $(v_xshock)"']
    for index in range(1, continuation_segments + 1):
        lines += [f"variable b{index} timer", "run 500", f"variable e{index} timer",
                  f'print "SEGMENT {index} $(v_e{index}-v_b{index}:%.9f)"',
                  f'print "OBS_SEG {index} $(step) $(atoms) $(temp) $(press) '
                  '$(density) $(etotal) $(v_xshock)"']
    return "\n".join(lines) + "\n"


def run_trial(output: Path, record: dict, params: Path, factor: float | None,
              repetition: int, continuation_segments: int) -> dict:
    label = "skip" if factor is None else f"factor-{factor:g}"
    key = f"phase-{record['phase']:02d}-{label}-r{repetition}"
    directory = output / "runs" / key
    directory.mkdir(parents=True, exist_ok=False)
    inp = directory / "in.lammps"
    inp.write_text(input_text(record, params, factor, continuation_segments))
    log = directory / "log.lammps"
    command = base.mpi_command(32, inp, log)
    command.insert(1, "--nooversubscribe")
    started = time.time()
    rc = None
    try:
        with (directory / "stdout.txt").open("w") as stdout, \
                (directory / "stderr.txt").open("w") as stderr:
            forks.CHILD = subprocess.Popen(
                command, cwd=directory, env={**os.environ, "OMP_NUM_THREADS": "1"},
                stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                rc = forks.CHILD.wait(timeout=360)
            except subprocess.TimeoutExpired:
                rc = -9
    finally:
        clean_child()
    text = log.read_text(errors="replace") if log.exists() else ""
    segment_rows = []
    for match in re.finditer(r"^SEGMENT (\d+) (.*)$", text, re.MULTILINE):
        values = [float(value) for value in match.group(2).split()]
        segment_rows.append({"index": int(match.group(1)), "times": values})
    observations = []
    for match in re.finditer(r"^OBS_SEG (\d+) (.*)$", text, re.MULTILINE):
        observations.append({"index": int(match.group(1)),
                             "values": [float(value) for value in match.group(2).split()]})
    blocks = [block for block in base.timing_blocks(text) if block["steps"] == 500]
    expected = continuation_segments + 1
    valid_times = (len(segment_rows) == expected and len(segment_rows[0]["times"]) == 4
                   and all(len(row["times"]) == (4 if row["index"] == 0 else 1)
                           for row in segment_rows))
    all_times = [value for row in segment_rows for value in row["times"]]
    safe = bool(rc == 0 and valid_times and len(observations) == expected
                and len(blocks) == expected
                and all(math.isfinite(value) and value >= 0 for value in all_times)
                and all(block["dangerous_builds"] == 0 for block in blocks)
                and int(observations[-1]["values"][0]) == int(record["step"]) + 500 * expected
                and not re.findall(r"^ERROR:.*$", text, re.MULTILINE))
    totals = [row["times"][0] for row in segment_rows] if valid_times else []
    row = {"key": key, "phase": record["phase"], "checkpoint_step": record["step"],
           "repetition": repetition, "factor": factor, "safe": safe, "returncode": rc,
           "segment_seconds": totals, "immediate_seconds": totals[0] if totals else None,
           "cumulative_seconds": sum(totals) if totals else None,
           "first_action_breakdown": segment_rows[0]["times"] if valid_times else None,
           "observations": observations, "timing_blocks": blocks,
           "timestamp": started, "process_wall_seconds": time.time()-started,
           "run_directory": str(directory)}
    (directory / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--continuation-segments", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    source = json.loads((args.checkpoints / "manifest.json").read_text())
    records = {int(row["phase"]): row for row in source["records"]}
    params = Path(source["initial_restart"]["path"]).parent / "shockparams.mod"
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = (output / "lock").open("w"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {"version":"shock-balance-persistence-v1", "actions":ACTIONS,
                "repetitions":args.repetitions, "continuation_segments":args.continuation_segments,
                "steps_per_segment":500, "mpi_ranks":32, "omp_threads":1,
                "source_sha256":digest(args.checkpoints / "manifest.json"),
                "common_continuation":"skip balance", "seed":args.seed}
    mp=output/"manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != json.loads(json.dumps(manifest)):
        raise RuntimeError("manifest mismatch")
    if not mp.exists(): mp.write_text(json.dumps(manifest,indent=2)+"\n")
    data=output/"measurements.jsonl"
    rows=[json.loads(line) for line in data.read_text().splitlines()] if data.exists() else []
    jobs=[(records[phase],factor,rep) for rep in range(args.repetitions)
          for phase,factors in ACTIONS.items() for factor in factors]
    rng=random.Random(args.seed)
    block=len(ACTIONS)*2
    chunks=[jobs[i:i+block] for i in range(0,len(jobs),block)]
    for chunk in chunks: rng.shuffle(chunk)
    jobs=[job for chunk in chunks for job in chunk]
    done={row["key"] for row in rows}
    with data.open("a") as stream:
        for record,factor,rep in jobs:
            label="skip" if factor is None else f"factor-{factor:g}"
            key=f"phase-{record['phase']:02d}-{label}-r{rep}"
            if key in done: continue
            row=run_trial(output,record,params,factor,rep,args.continuation_segments)
            stream.write(json.dumps(row)+"\n");stream.flush();os.fsync(stream.fileno())
            print(key,"safe",row["safe"],"immediate",row["immediate_seconds"],
                  "cumulative",row["cumulative_seconds"],flush=True)
            if not row["safe"]: raise RuntimeError(f"invalid trial {key}")
            done.add(key)
    (output/"complete.json").write_text(json.dumps({"trials":len(jobs),"timestamp":time.time()},indent=2)+"\n")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try: main()
    finally: clean_child()
