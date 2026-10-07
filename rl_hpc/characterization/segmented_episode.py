#!/usr/bin/env python3
"""Run a fixed-action LAMMPS episode while external contention changes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

from characterize import (
    CHAR_DIR,
    ROOT,
    WORKLOADS,
    Action,
    parse_log,
    start_contender,
    stop_contender,
    system_snapshot,
)


_LMP_CLASS = None
_LMP_TEMP = None


def load_lammps(build_dir: Path):
    global _LMP_CLASS, _LMP_TEMP
    if _LMP_CLASS is not None:
        return _LMP_CLASS, _LMP_TEMP
    library = build_dir / "liblammps.so"
    if not library.exists():
        raise RuntimeError(f"LAMMPS shared library not found: {library}")
    package_source = ROOT / "python" / "lammps"
    temp_dir = tempfile.TemporaryDirectory(prefix="lammps-characterization-")
    package_dir = Path(temp_dir.name) / "lammps"
    shutil.copytree(package_source, package_dir)
    (package_dir / "liblammps.so").symlink_to(library)
    sys.path.insert(0, temp_dir.name)
    from lammps import lammps  # pylint: disable=import-outside-toplevel
    _LMP_CLASS = lammps
    _LMP_TEMP = temp_dir
    return _LMP_CLASS, _LMP_TEMP


def setup_lines(workload: dict, action: Action) -> list[str]:
    input_path = Path(workload["input"])
    skip_names = {"steps", "seed", "skin", "neigh_every", "neigh_delay", "neigh_check", "datafile"}
    lines: list[str] = []
    variables = {"steps": "1", "seed": str(workload["seed"]), **action.variables(), **workload.get("extra_variables", {})}
    for name, value in variables.items():
        lines.append(f"variable {name} index {value}")
    for line in input_path.read_text().splitlines():
        match = re.match(r"\s*variable\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if match and match.group(1) in skip_names:
            continue
        if re.match(r"\s*run\s+", line):
            continue
        lines.append(line)
    return lines


def run_episode(workload_name: str, phase_order: list[str], segment_steps: int, output: Path) -> list[dict]:
    workload = WORKLOADS[workload_name]
    action = workload["actions"][0]
    episode_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{workload_name}-{'-'.join(phase_order)}"
    episode_dir = output / episode_id
    episode_dir.mkdir(parents=True)
    log_path = episode_dir / "log.lammps"
    lammps_class, temp_dir = load_lammps(ROOT / "build_rl_char")
    records: list[dict] = []
    original_affinity = None
    try:
        if hasattr(os, "sched_getaffinity"):
            original_affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {0, 1, 2, 3})
        lmp = lammps_class(cmdargs=["-log", str(log_path), "-screen", "none"])
        try:
            lmp.commands_list(setup_lines(workload, action))
            for segment_id, state in enumerate(phase_order):
                contender = None
                before = system_snapshot()
                started = time.perf_counter()
                try:
                    contender = start_contender(state, 20.0, None)
                    if contender is not None:
                        time.sleep(0.4)
                    lmp.command(f"run {segment_steps}")
                finally:
                    stop_contender(contender)
                wall = time.perf_counter() - started
                after = system_snapshot()
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "episode_id": episode_id,
                    "segment_id": segment_id,
                    "workload": workload_name,
                    "backend": "cpu",
                    "contention_state": state,
                    "action": {"name": action.name, **action.variables()},
                    "segment_steps": segment_steps,
                    "segment_wall_seconds_python": wall,
                    "thermo": {
                        key: lmp.get_thermo(key)
                        for key in ("step", "temp", "cpu", "spcpu")
                        if _thermo_available(lmp, key)
                    },
                    "lammps": parse_log(log_path),
                    "system_before": before,
                    "system_after": after,
                    "log": str(log_path.relative_to(ROOT)),
                }
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
        finally:
            lmp.close()
    finally:
        if original_affinity is not None:
            os.sched_setaffinity(0, original_affinity)
    return records


def _thermo_available(lmp: object, name: str) -> bool:
    try:
        value = lmp.get_thermo(name)
    except Exception:  # LAMMPS reports unavailable thermo keywords by exception.
        return False
    return isinstance(value, (float, int))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CHAR_DIR / "data" / "episodes")
    parser.add_argument("--segment-steps", type=int, default=20)
    parser.add_argument(
        "--idle-only",
        action="store_true",
        help="run four consecutive idle segments per workload for an internal-phase probe",
    )
    args = parser.parse_args()
    if args.segment_steps < 1:
        parser.error("--segment-steps must be positive")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    phase_orders = (
        {"episode_idle": ["idle", "idle", "idle", "idle"]}
        if args.idle_only
        else {
            "episode_a": ["idle", "cpu_contention", "memory_contention", "idle"],
            "episode_b": ["memory_contention", "idle", "cpu_contention", "idle"],
        }
    )
    all_records: list[dict] = []
    try:
        for workload_name in ("lj", "spce"):
            for order in phase_orders.values():
                all_records.extend(run_episode(workload_name, order, args.segment_steps, output))
    finally:
        if _LMP_TEMP is not None:
            _LMP_TEMP.cleanup()
    path = output / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in all_records) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(phase_orders) * 2, "segments": len(all_records), "output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
