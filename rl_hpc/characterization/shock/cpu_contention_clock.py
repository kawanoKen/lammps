#!/usr/bin/env python3
"""Wall-clock CPU contention independent of LAMMPS decision boundaries."""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import time

import psutil

import cpu_jitter


def make_trace(seed: int, minimum: float, maximum: float,
               horizon: float = 7200.0) -> list[dict]:
    if minimum <= 0 or maximum < minimum or horizon <= maximum:
        raise ValueError("invalid wall-clock contention schedule")
    rng = random.Random(seed)
    active = bool(rng.randrange(2))
    elapsed = 0.0
    trace = []
    while elapsed < horizon:
        duration = rng.uniform(minimum, maximum)
        trace.append({"start": elapsed, "end": elapsed + duration,
                      "level": "L4" if active else "L0", "active": active})
        elapsed += duration
        active = not active
    return trace


def _scheduler(trace: list[dict], epoch: float, active: mp.synchronize.Event,
               stop: mp.synchronize.Event, level: mp.sharedctypes.Synchronized,
               ready: mp.synchronize.Event) -> None:
    ready.set()
    for block in trace:
        if stop.is_set():
            return
        level.value = 4 if block["active"] else 0
        if block["active"]:
            active.set()
        else:
            active.clear()
        deadline = epoch + block["end"]
        while not stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            stop.wait(min(remaining, 0.2))
    active.clear()


class WallClockContention:
    def __init__(self, cores: tuple[int, ...], seed: int,
                 minimum: float = 45.0, maximum: float = 90.0,
                 horizon: float = 7200.0):
        self.cores = cores
        self.seed = seed
        self.minimum = minimum
        self.maximum = maximum
        self.trace = make_trace(seed, minimum, maximum, horizon)
        context = mp.get_context("spawn")
        self.active = context.Event()
        self.stop_event = context.Event()
        self.level = context.Value("i", -1)
        self.worker_ready = [context.Event() for _ in cores]
        self.scheduler_ready = context.Event()
        self.workers = [context.Process(
            target=cpu_jitter._worker,
            args=(core, self.active, self.stop_event, ready), daemon=False)
            for core, ready in zip(cores, self.worker_ready)]
        self.epoch = 0.0
        self.scheduler = None

    def start(self) -> None:
        try:
            for worker in self.workers:
                worker.start()
            if not all(event.wait(timeout=15) for event in self.worker_ready):
                raise RuntimeError("CPU contention worker failed to initialize")
            for core, worker in zip(self.cores, self.workers):
                process = psutil.Process(worker.pid)
                if worker.exitcode is not None or process.cpu_affinity() != [core]:
                    raise RuntimeError("CPU contention worker affinity mismatch")
            self.epoch = time.monotonic()
            context = mp.get_context("spawn")
            self.scheduler = context.Process(
                target=_scheduler,
                args=(self.trace, self.epoch, self.active, self.stop_event,
                      self.level, self.scheduler_ready), daemon=False)
            self.scheduler.start()
            if not self.scheduler_ready.wait(timeout=15):
                raise RuntimeError("CPU contention scheduler failed to initialize")
        except BaseException:
            self.close()
            raise

    def _assert_alive(self) -> None:
        processes = self.workers + ([self.scheduler] if self.scheduler else [])
        if any(process.exitcode is not None for process in processes):
            raise RuntimeError("CPU contention child exited unexpectedly")

    def block_at(self, timestamp: float) -> dict:
        elapsed = max(0.0, timestamp - self.epoch)
        for block in self.trace:
            if block["start"] <= elapsed < block["end"]:
                return block
        raise RuntimeError("contention trace horizon exhausted")

    def residency(self, begin: float, end: float) -> dict:
        if end < begin:
            raise ValueError("reversed interval")
        self._assert_alive()
        lo, hi = begin - self.epoch, end - self.epoch
        l0 = l4 = 0.0
        for block in self.trace:
            overlap = max(0.0, min(hi, block["end"]) - max(lo, block["start"]))
            if block["active"]:
                l4 += overlap
            else:
                l0 += overlap
            if block["start"] > hi:
                break
        return {"l0_seconds": l0, "l4_seconds": l4,
                "mixed": l0 > 1e-9 and l4 > 1e-9,
                "start_level": self.block_at(begin)["level"],
                "end_level": self.block_at(max(begin, end - 1e-9))["level"]}

    def cpu_seconds(self) -> float:
        self._assert_alive()
        total = 0.0
        for worker in self.workers:
            times = psutil.Process(worker.pid).cpu_times()
            total += times.user + times.system
        return total

    def close(self) -> None:
        self.active.clear()
        self.stop_event.set()
        processes = self.workers + ([self.scheduler] if self.scheduler else [])
        for process in processes:
            if process.pid is None:
                continue
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)


def main() -> None:
    cores = cpu_jitter.parse_cores("0-7")
    contender = WallClockContention(cores, seed=20261014, minimum=1, maximum=2,
                                    horizon=30)
    contender.start()
    try:
        start = time.monotonic()
        time.sleep(5)
        end = time.monotonic()
        print(contender.residency(start, end), contender.cpu_seconds())
    finally:
        contender.close()


if __name__ == "__main__":
    main()
