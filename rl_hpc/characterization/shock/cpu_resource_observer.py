#!/usr/bin/env python3
"""Non-blocking CPU counter snapshots for a persistent MPI LAMMPS job."""
from __future__ import annotations

from dataclasses import dataclass
import time

import psutil


@dataclass(frozen=True)
class Snapshot:
    monotonic: float
    cpu_times: tuple
    lammps_cpu_seconds: float


def _process_tree_cpu_seconds(root_pid: int) -> float:
    try:
        root = psutil.Process(root_pid)
        processes = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess as exc:
        raise RuntimeError("LAMMPS process tree disappeared") from exc
    total = 0.0
    for process in processes:
        try:
            value = process.cpu_times()
            total += value.user + value.system
        except psutil.NoSuchProcess:
            continue
    return total


def wait_for_mpi_tree(root_pid: int, minimum_processes: int = 32,
                      timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            children = psutil.Process(root_pid).children(recursive=True)
        except psutil.NoSuchProcess as exc:
            raise RuntimeError("MPI launcher exited during initialization") from exc
        # The launcher itself may replace one rank or retain all ranks as
        # children depending on the Open MPI startup path. Count the root.
        if 1 + len(children) >= minimum_processes:
            return
        time.sleep(0.02)
    raise RuntimeError(f"expected at least {minimum_processes} MPI processes")


class CpuResourceObserver:
    def __init__(self, target_cores: tuple[int, ...], mpi_ranks: int = 32):
        self.target_cores = target_cores
        self.mpi_ranks = mpi_ranks

    def snapshot(self, root_pid: int) -> Snapshot:
        return Snapshot(time.monotonic(), tuple(psutil.cpu_times(percpu=True)),
                        _process_tree_cpu_seconds(root_pid))

    def summarize(self, before: Snapshot, after: Snapshot) -> dict:
        wall = after.monotonic - before.monotonic
        if wall <= 0:
            raise RuntimeError("non-positive telemetry window")
        values = []
        for core in self.target_cores:
            first, last = before.cpu_times[core], after.cpu_times[core]
            fields = first._fields
            total = sum(getattr(last, field) - getattr(first, field) for field in fields)
            idle = (last.idle - first.idle)
            if "iowait" in fields:
                idle += last.iowait - first.iowait
            values.append(100.0 * max(0.0, min(1.0, 1.0 - idle / max(total, 1e-12))))
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        lammps_delta = max(0.0, after.lammps_cpu_seconds - before.lammps_cpu_seconds)
        return {
            "window_seconds": wall,
            "target_cpu_util_mean": mean,
            "target_cpu_util_max": max(values),
            "target_cpu_util_std": variance ** 0.5,
            "target_cpu_util_per_core": values,
            "lammps_cpu_seconds": lammps_delta,
            "lammps_cpu_entitlement": lammps_delta / (wall * self.mpi_ranks),
        }
