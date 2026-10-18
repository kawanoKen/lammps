#!/usr/bin/env python3
"""Pre-episode cleanliness guard and low-rate host process monitoring."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

import psutil


def _cpu_utilization(before: tuple, after: tuple) -> list[float]:
    values = []
    for first, last in zip(before, after):
        fields = first._fields
        total = sum(getattr(last, field) - getattr(first, field)
                    for field in fields)
        idle = last.idle - first.idle
        if "iowait" in fields:
            idle += last.iowait - first.iowait
        values.append(100.0 * max(0.0, min(1.0,
            1.0 - idle / max(total, 1e-12))))
    return values


def _process_snapshot() -> dict[int, tuple[float, dict]]:
    rows = {}
    for process in psutil.process_iter(["pid", "ppid", "name", "username",
                                        "cmdline"]):
        try:
            times = process.cpu_times()
            info = dict(process.info)
            info["cmdline"] = " ".join(info.get("cmdline") or [])[:300]
            info["cpu_affinity"] = process.cpu_affinity()
            rows[process.pid] = (times.user + times.system, info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return rows


def _is_descendant(pid: int, root_pid: int) -> bool:
    if pid == root_pid:
        return True
    try:
        process = psutil.Process(pid)
        return any(parent.pid == root_pid for parent in process.parents())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def sample(window_seconds: float, target_cores: tuple[int, ...],
           owner_pid: int) -> dict:
    cpu_before = tuple(psutil.cpu_times(percpu=True))
    process_before = _process_snapshot()
    begin = time.monotonic()
    time.sleep(window_seconds)
    end = time.monotonic()
    cpu_after = tuple(psutil.cpu_times(percpu=True))
    process_after = _process_snapshot()
    wall = end - begin
    utilization = _cpu_utilization(cpu_before, cpu_after)
    top = []
    for pid, (cpu_after_seconds, info) in process_after.items():
        previous = process_before.get(pid)
        if previous is None:
            continue
        percent = 100.0 * max(0.0, cpu_after_seconds - previous[0]) / wall
        if percent < 1.0:
            continue
        affinity = set(info.get("cpu_affinity") or [])
        top.append({
            "pid": pid, "ppid": info.get("ppid"),
            "name": info.get("name"), "username": info.get("username"),
            "cmdline": info.get("cmdline"), "cpu_percent": percent,
            "owned_by_study": _is_descendant(pid, owner_pid),
            "target_affinity_overlap": bool(affinity.intersection(target_cores)),
        })
    target = [utilization[core] for core in target_cores]
    return {
        "timestamp": time.time(), "window_seconds": wall,
        "target_cpu_util_mean": sum(target) / len(target),
        "target_cpu_util_max": max(target),
        "target_cpu_util_per_core": target,
        "system_cpu_util_mean": sum(utilization) / len(utilization),
        "load_average": list(os.getloadavg()),
        "top_processes": sorted(top, key=lambda row: row["cpu_percent"],
                                reverse=True)[:20],
    }


def wait_for_clean_baseline(output: Path, target_cores: tuple[int, ...],
                            *, window_seconds: float = 5.0,
                            consecutive: int = 2,
                            target_mean_limit: float = 15.0,
                            target_max_limit: float = 50.0,
                            system_mean_limit: float = 15.0,
                            load1_limit: float = 8.0,
                            timeout_seconds: float = 1800.0) -> list[dict]:
    deadline = time.monotonic() + timeout_seconds
    clean = 0
    attempts = []
    while time.monotonic() < deadline:
        row = sample(window_seconds, target_cores, os.getpid())
        row["clean"] = bool(
            row["target_cpu_util_mean"] <= target_mean_limit
            and row["target_cpu_util_max"] <= target_max_limit
            and row["system_cpu_util_mean"] <= system_mean_limit
            and row["load_average"][0] <= load1_limit)
        attempts.append(row)
        output.write_text(json.dumps({
            "thresholds": {
                "window_seconds": window_seconds, "consecutive": consecutive,
                "target_mean_limit": target_mean_limit,
                "target_max_limit": target_max_limit,
                "system_mean_limit": system_mean_limit,
                "load1_limit": load1_limit,
                "timeout_seconds": timeout_seconds,
            },
            "attempts": attempts, "accepted": False,
        }, indent=2) + "\n")
        clean = clean + 1 if row["clean"] else 0
        if clean >= consecutive:
            payload = json.loads(output.read_text())
            payload["accepted"] = True
            output.write_text(json.dumps(payload, indent=2) + "\n")
            return attempts
        time.sleep(5.0)
    raise RuntimeError("host did not reach a clean CPU baseline within timeout")


class HostProcessMonitor:
    def __init__(self, output: Path, target_cores: tuple[int, ...],
                 interval_seconds: float = 5.0):
        self.output = output
        self.target_cores = target_cores
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        with self.output.open("a") as stream:
            while not self.stop_event.is_set():
                row = sample(self.interval_seconds, self.target_cores,
                             os.getpid())
                stream.write(json.dumps(row) + "\n")
                stream.flush()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval_seconds + 5.0)
