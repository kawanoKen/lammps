#!/usr/bin/env python3
"""Rank-targeted x-slice balance for the existing MPI32 Shock/NEMD setup.

The action is a vector of positive rank capacities.  Like LAMMPS's
``balance shift x``, this finds x cuts at cumulative load targets; unlike the
built-in equal-target heuristic, the target intervals can differ by rank.
Only the existing ``balance x`` command changes the LAMMPS partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

import collect_trajectories as collector
import cpu_jitter
import shock_counterfactual as base


P = 32
ATOMS = 491520
STEPS = 500
SKIN = 0.5
CORES = tuple(range(8))
CAPACITY = tuple(0.5 if rank in CORES else 1.0 for rank in range(P))
POLICIES = ("skip", "equal_shift", "target_vector")
REGIMES = ("idle", "contention")


def target_counts(total: int, capacities: tuple[float, ...]) -> np.ndarray:
    values = np.asarray(capacities, dtype=float)
    if len(values) != P or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("expected 32 finite, positive rank capacities")
    return total * values / values.sum()


def cuts_from_x(x: np.ndarray, xlo: float, xhi: float,
                capacities: tuple[float, ...], skin: float = SKIN) -> np.ndarray:
    """Invert the empirical x CDF at cumulative capacity fractions.

    A cut is placed midway between distinct adjacent x planes.  When a
    target falls within an unsplittable lattice plane, choose the nearest
    whole-plane boundary.  This mirrors the geometric limit documented for
    LAMMPS's own grid balancer.  Reject skin-unsafe narrow domains.
    """
    if xhi <= xlo or len(x) < P or not np.all(np.isfinite(x)):
        raise ValueError("invalid atom coordinates or box")
    x = np.sort(np.asarray(x, dtype=float))
    if x[0] < xlo or x[-1] >= xhi:
        raise ValueError("atom outside x box")
    desired = np.cumsum(target_counts(len(x), capacities))[:-1]
    unique, multiplicity = np.unique(x, return_counts=True)
    cumulative = np.cumsum(multiplicity)[:-1]
    if len(cumulative) < P - 1:
        raise ValueError("too few distinct x planes")
    boundaries = (unique[:-1] + unique[1:]) / 2
    cuts = []
    previous = -1
    for k, wanted in enumerate(desired):
        index = int(np.searchsorted(cumulative, wanted))
        candidates = [j for j in (index - 1, index) if previous < j < len(cumulative)]
        if not candidates:
            raise ValueError(f"cannot assign distinct cut {k + 1}")
        chosen = min(candidates, key=lambda j: (abs(cumulative[j] - wanted), j))
        cuts.append(boundaries[chosen])
        previous = chosen
    cuts = np.asarray(cuts)
    fractions = (cuts - xlo) / (xhi - xlo)
    if np.any(np.diff(np.r_[0.0, cuts, xhi]) < skin):
        raise ValueError("target requires a subdomain narrower than neighbor skin")
    if np.any(np.diff(fractions) <= 0) or not 0 < fractions[0] < fractions[-1] < 1:
        raise ValueError("invalid x cuts")
    return fractions


def read_dump(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    with path.open() as stream:
        if stream.readline().strip() != "ITEM: TIMESTEP":
            raise ValueError("not a LAMMPS custom dump")
        stream.readline()
        if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError("missing atom count")
        total = int(stream.readline())
        if total != ATOMS:
            raise ValueError(f"atom count {total} != {ATOMS}")
        if not stream.readline().startswith("ITEM: BOX BOUNDS"):
            raise ValueError("missing box bounds")
        xlo, xhi = map(float, stream.readline().split()[:2])
        stream.readline()
        stream.readline()
        if stream.readline().strip() != "ITEM: ATOMS id x proc":
            raise ValueError("unexpected dump columns")
        ids = set()
        x = np.empty(total)
        ranks = np.empty(total, dtype=np.int32)
        for index, line in enumerate(stream):
            fields = line.split()
            if len(fields) != 3 or index >= total:
                raise ValueError("malformed atom row")
            atom_id, coord, rank = int(fields[0]), float(fields[1]), int(fields[2])
            if atom_id in ids or not 0 <= rank < P:
                raise ValueError("duplicate atom ID or invalid rank")
            ids.add(atom_id)
            x[index], ranks[index] = coord, rank
        if len(ids) != total:
            raise ValueError("incomplete atom dump")
    return x, np.bincount(ranks, minlength=P), xlo, xhi


def checked_state(text: str, previous_step: int | None = None) -> dict:
    state = collector.observation(text, SKIN, 1.0, expected_atoms=ATOMS)
    if previous_step is not None and int(state["step"]) != previous_step + STEPS:
        raise RuntimeError("incorrect 500-step increment")
    if state["previous_segment"]["dangerous_builds"]:
        raise RuntimeError("dangerous neighbor build")
    return state


def dump_counts(sim: collector.Simulation, directory: Path, label: str):
    path = directory / f"{label}.dump"
    sim.execute(f"write_dump all custom {path} id x proc\n")
    try:
        return read_dump(path)
    finally:
        path.unlink(missing_ok=True)


def run_one(root: Path, restart: Path, state_label: str, policy: str,
            regime: str, repeat: int, action_capacity: tuple[float, ...]) -> dict:
    name = f"{state_label}-{regime}-{policy}-r{repeat}"
    directory = root / name
    directory.mkdir()
    jitter = cpu_jitter.CpuJitter(CORES) if regime == "contention" else None
    sim = None
    try:
        if jitter:
            jitter.start()
            jitter.set_active(True)
        sim = collector.Simulation(directory)
        params = base.ROOT / "rl_hpc/characterization/data/portable_shock_initial_v1/shockparams.mod"
        setup = (f"include {params}\n"
                 "processors 32 1 1 map xyz\n"
                 f"read_restart {restart}\n"
                 "fix mynve all nve\n"
                 "fix xwalls all wall/reflect xlo EDGE xhi EDGE\n"
                 "compute mymom all momentum\n"
                 "variable xshock equal (1.0+c_mymom[1]/(atoms*${mass})/${up})*lx-${up}*time\n"
                 "timer normal\nneighbor 0.5 bin\n"
                 "neigh_modify every 1 delay 0 check yes\n"
                 "thermo 100\n"
                 "thermo_style custom step atoms temp pe density etotal press v_xshock\n"
                 "run 0 post no\n"
                 "balance 1.0 shift x 10 1.0 weight neigh 1.0\n"
                 "run 50\n" + collector.OBS + "\n")
        before = checked_state(sim.execute(setup))
        control_begin = time.perf_counter()
        x, initial, xlo, xhi = dump_counts(sim, directory, "before")
        if int(initial.sum()) != ATOMS:
            raise RuntimeError("lost atoms before action")
        capacities = action_capacity if policy == "target_vector" else (1.0,) * P
        target = target_counts(ATOMS, capacities)
        if policy == "target_vector":
            cuts = cuts_from_x(x, xlo, xhi, capacities)
            command = "balance 0.0 x " + " ".join(f"{v:.15g}" for v in cuts)
        elif policy == "equal_shift":
            cuts = None
            command = "balance 1.0 shift x 10 1.0"
        else:
            cuts = None
            command = ""
        controller_seconds = (time.perf_counter() - control_begin
                              if policy == "target_vector" else 0.0)
        action_output = sim.execute(
            "variable action_begin timer\n" + (command + "\n" if command else "") +
            "variable action_end timer\n"
            'print "ACTION_SECONDS $(v_action_end-v_action_begin:%.9f)"\n')
        if "ERROR" in action_output or "nan" in action_output.lower():
            raise RuntimeError("unsafe action output")
        import re
        action_match = re.findall(r"^ACTION_SECONDS ([0-9.eE+-]+)$", action_output, re.M)
        if not action_match:
            raise RuntimeError("missing action time")
        action_seconds = float(action_match[-1])
        if not math.isfinite(action_seconds) or action_seconds < 0:
            raise RuntimeError("invalid action time")
        _, applied, _, _ = dump_counts(sim, directory, "applied")
        final_text = sim.execute("run 500\n" + collector.OBS + "\n")
        after = checked_state(final_text, int(before["step"]))
        _, after_counts, _, _ = dump_counts(sim, directory, "after")
        blocks = base.timing_blocks(final_text)
        block = next((b for b in reversed(blocks) if b["steps"] == STEPS), None)
        if block is None:
            raise RuntimeError("missing 500-step timing")
        md_seconds = float(block["loop_seconds"])
        if not math.isfinite(md_seconds) or md_seconds <= 0:
            raise RuntimeError("invalid MD time")
        code = sim.close(graceful=True)
        sim = None
        if code != 0:
            raise RuntimeError(f"MPI exit code {code}")
        return dict(state=state_label, regime=regime, policy=policy, repeat=repeat,
                    step_before=int(before["step"]), step_after=int(after["step"]),
                    capacity=list(capacities), target=target.tolist(),
                    initial=initial.tolist(), applied=applied.tolist(),
                    after=after_counts.tolist(), cuts=cuts.tolist() if cuts is not None else None,
                    action_seconds=action_seconds, md_seconds=md_seconds,
                    total_seconds=action_seconds + md_seconds,
                    controller_seconds=controller_seconds,
                    end_to_end_seconds=controller_seconds + action_seconds + md_seconds,
                    dangerous_builds=0, mpi_returncode=code)
    finally:
        if sim is not None:
            sim.close()
        if jitter is not None:
            jitter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--states", nargs="+", choices=("S1", "S5", "S8"),
                        default=("S1", "S5", "S8"))
    parser.add_argument("--policies", nargs="+", choices=POLICIES,
                        default=POLICIES)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES,
                        default=REGIMES)
    parser.add_argument("--capacity-vector", default=",".join(map(str, CAPACITY)),
                        help="32 comma-separated positive target-load shares, in rank order")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not set(CORES).issubset(os.sched_getaffinity(0)):
        parser.error("contention CPU cores are unavailable")
    try:
        action_capacity = tuple(float(value) for value in args.capacity_vector.split(","))
        target_counts(ATOMS, action_capacity)
    except ValueError as exc:
        parser.error(f"invalid capacity vector: {exc}")
    restart_root = base.ROOT / "rl_hpc/characterization/data/portable_representative_checkpoints_v1"
    paths = {"S1": restart_root / "phase-01-step-4800.restart",
             "S5": restart_root / "phase-05-step-35800.restart",
             "S8": restart_root / "phase-08-step-57300.restart"}
    if any(not paths[name].is_file() for name in args.states):
        parser.error("representative restart missing")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = [(state, regime, policy, rep) for state in args.states
            for regime in args.regimes for rep in range(args.repeats)
            for policy in args.policies]
    random.Random(20261014).shuffle(plan)
    manifest = dict(status="running", git_commit=base.git_commit(),
                    restart_sha256={s: hashlib.sha256(paths[s].read_bytes()).hexdigest()
                                    for s in args.states},
                    mpi_ranks=P, omp_threads=1, atoms=ATOMS, steps=STEPS,
                    contention_cores=list(CORES), capacity=list(action_capacity),
                    trial_order=plan, completed=0)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    try:
        for index, (state, regime, policy, rep) in enumerate(plan, 1):
            print(f"TRIAL {index}/{len(plan)} {state} {regime} {policy} r{rep}", flush=True)
            row = run_one(output, paths[state], state, policy, regime, rep,
                          action_capacity)
            rows.append(row)
            with (output / "trials.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            manifest["completed"] = index
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "stopped"
        manifest["error"] = repr(exc)
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
