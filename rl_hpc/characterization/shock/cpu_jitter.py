#!/usr/bin/env python3
"""Pinned, switchable CPU co-runner for the MPI32 shock experiment.

One worker is pinned to each requested physical CPU. The parent toggles a
shared event only at LAMMPS decision boundaries. No privileged controls or
global machine configuration are changed.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import time

import psutil


def parse_cores(spec: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in spec.split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-', 1))
            if lo > hi:
                raise ValueError('reversed CPU range')
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    if not values or len(values) != len(set(values)):
        raise ValueError('CPU list must be nonempty and unique')
    allowed = os.sched_getaffinity(0)
    if not set(values).issubset(allowed):
        raise ValueError('requested CPU is outside allowed affinity')
    return tuple(values)


def schedule(horizon: int, seed: int, min_length: int = 2,
             max_length: int = 5) -> list[bool]:
    if horizon < 1 or min_length < 1 or max_length < min_length:
        raise ValueError('invalid schedule length')
    rng = random.Random(seed)
    active = bool(rng.randrange(2))
    values: list[bool] = []
    while len(values) < horizon:
        values.extend([active] * rng.randint(min_length, max_length))
        active = not active
    return values[:horizon]


def _worker(core: int, active: mp.synchronize.Event,
            stop: mp.synchronize.Event, ready: mp.synchronize.Event) -> None:
    os.sched_setaffinity(0, {core})
    ready.set()
    value = 0x12345678 ^ core
    while not stop.is_set():
        if not active.is_set():
            time.sleep(.02)
            continue
        for _ in range(4096):
            value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
            value ^= value >> 13
    if value == -1:
        print('unreachable', flush=True)


class CpuJitter:
    def __init__(self, cores: tuple[int, ...]):
        self.cores = cores
        ctx = mp.get_context('spawn')
        self.active = ctx.Event()
        self.stop_event = ctx.Event()
        self.ready = [ctx.Event() for _ in cores]
        self.children = [ctx.Process(target=_worker, args=(core, self.active,
                         self.stop_event, ready), daemon=False)
                         for core, ready in zip(cores, self.ready)]
        self.is_active = False

    def start(self) -> None:
        try:
            for child in self.children:
                child.start()
            if not all(ready.wait(timeout=15) for ready in self.ready):
                raise RuntimeError('CPU co-runner worker failed to initialize')
            for core, child in zip(self.cores, self.children):
                if child.exitcode is not None or psutil.Process(child.pid).cpu_affinity() != [core]:
                    raise RuntimeError(f'CPU co-runner affinity mismatch on core {core}')
        except BaseException:
            self.close()
            raise

    def set_active(self, value: bool) -> None:
        if any(child.exitcode is not None for child in self.children):
            raise RuntimeError('CPU co-runner worker exited')
        if value:
            self.active.set()
        else:
            self.active.clear()
        self.is_active = value

    def cpu_seconds(self) -> float:
        total = 0.0
        for child in self.children:
            if child.pid is None:
                continue
            try:
                times = psutil.Process(child.pid).cpu_times()
            except psutil.NoSuchProcess as exc:
                raise RuntimeError('CPU co-runner worker disappeared') from exc
            total += times.user + times.system
        return total

    def close(self) -> None:
        self.active.clear()
        self.stop_event.set()
        for child in self.children:
            if child.pid is None:
                continue
            child.join(timeout=3)
            if child.is_alive():
                child.terminate()
                child.join(timeout=3)
            if child.is_alive():
                child.kill()
                child.join(timeout=3)

