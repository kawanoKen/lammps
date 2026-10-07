#!/usr/bin/env python3
"""Small auditable co-runners for controlled resource contention."""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import signal
import time


STOP = False


def stop_handler(signum: int, frame: object) -> None:
    del signum, frame
    global STOP
    STOP = True


def cpu_worker(deadline: float) -> None:
    value = 0x12345678
    while time.monotonic() < deadline and not STOP:
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
        value ^= value >> 13
    if value == 0:
        print("unreachable", flush=True)


def run_cpu(duration: float, workers: int) -> None:
    deadline = time.monotonic() + duration
    children = [mp.Process(target=cpu_worker, args=(deadline,)) for _ in range(workers)]
    for child in children:
        child.start()
    try:
        while time.monotonic() < deadline and not STOP:
            time.sleep(0.1)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=2)


def run_memory(duration: float, size_mb: int) -> None:
    size = size_mb * 1024 * 1024
    source = ctypes.create_string_buffer(size)
    target = ctypes.create_string_buffer(size)
    libc = ctypes.CDLL(None)
    libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    source_ptr = ctypes.cast(source, ctypes.c_void_p)
    target_ptr = ctypes.cast(target, ctypes.c_void_p)
    target_bytes = ctypes.cast(target, ctypes.POINTER(ctypes.c_ubyte))
    deadline = time.monotonic() + duration
    checksum = 0
    while time.monotonic() < deadline and not STOP:
        libc.memset(source_ptr, checksum & 0xFF, size)
        libc.memcpy(target_ptr, source_ptr, size)
        checksum = (checksum + int(target_bytes[0]) + int(target_bytes[size - 1]) + 1) & 0xFFFFFFFF
    if checksum == -1:
        print("unreachable", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "memory"))
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size-mb", type=int, default=512)
    args = parser.parse_args()
    if args.duration <= 0 or args.workers < 1 or args.size_mb < 64:
        parser.error("duration/workers must be positive and size-mb must be >= 64")
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    if args.mode == "cpu":
        run_cpu(args.duration, args.workers)
    else:
        run_memory(args.duration, args.size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
