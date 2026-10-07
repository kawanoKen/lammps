#!/usr/bin/env python3
"""Independent-process GPU contention measurements with thermal/power audit."""
from __future__ import annotations

import argparse, json, os, re, shlex, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

from characterize import ROOT, WORKLOAD_DIR, parse_log, git_commit
from gpu_factorial import build_gpu_contender

TARGET_GPU = "0"
CONTENDER_GPU = {"A": None, "B": "0", "C": "1", "D": "3"}
INTENSITY = (128, 32, 0)  # MiB per array, arithmetic repeats, sleep us


def smi() -> list[dict]:
    q = "index,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used"
    try:
        text = subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], text=True, timeout=4)
    except (OSError, subprocess.SubprocessError):
        return []
    names = ("index", "temperature_c", "power_w", "clock_sm_mhz", "clock_mem_mhz", "utilization_gpu_percent", "utilization_memory_percent", "memory_used_mb")
    rows = []
    for line in text.splitlines():
        values = [x.strip() for x in line.split(",")]
        if len(values) == len(names):
            try: rows.append(dict(zip(names, [int(values[0])] + [float(x) for x in values[1:]])))
            except ValueError: pass
    return rows


def target_row(rows: list[dict]) -> dict | None:
    return next((r for r in rows if r["index"] == 0), None)


def cool(max_temp: float, max_power: float, timeout: float) -> list[dict]:
    end = time.monotonic() + timeout
    while True:
        rows = smi(); row = target_row(rows)
        if row and row["temperature_c"] <= max_temp and row["power_w"] <= max_power and row["utilization_gpu_percent"] <= 2:
            return rows
        if time.monotonic() >= end:
            raise RuntimeError(f"GPU0 did not cool below {max_temp} C / {max_power} W within {timeout} s: {row}")
        time.sleep(5)


def stop(proc: subprocess.Popen | None) -> tuple[int | None, str]:
    if not proc: return None, ""
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
    try: out, _ = proc.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL); out, _ = proc.communicate(timeout=8)
    return proc.returncode, out or ""


def phase(root: Path, phase_id: str, state: str, skin: str, every: int, steps: int, lmp: Path, contender: Path, period: float, cool_temp: float, cool_power: float, cool_timeout: float) -> dict:
    pre = cool(cool_temp, cool_power, cool_timeout)
    directory = root / phase_id; directory.mkdir(parents=True)
    log = directory / "log.lammps"
    cmd = ["taskset", "-c", "0-3", str(lmp), "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "gpu/aware", "off", "-log", str(log), "-screen", "none", "-var", "steps", str(steps), "-var", "seed", "87287", "-var", "skin", skin, "-var", "neigh_every", str(every), "-var", "neigh_delay", "0", "-var", "neigh_check", "yes", "-in", str(WORKLOAD_DIR / "lj_scaled.in")]
    (directory / "command.txt").write_text(shlex.join(cmd) + "\n")
    bg = None; samples = []; started = time.monotonic()
    try:
        if CONTENDER_GPU[state] is not None:
            mb, repeats, sleep_us = INTENSITY
            bg = subprocess.Popen(["taskset", "-c", "8-11", str(contender), "90", str(mb), str(repeats), str(sleep_us)], env={**os.environ, "CUDA_VISIBLE_DEVICES": CONTENDER_GPU[state]}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            time.sleep(0.7)
        target = subprocess.Popen(cmd, cwd=directory, env={**os.environ, "CUDA_VISIBLE_DEVICES": TARGET_GPU, "OMP_NUM_THREADS": "1"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        while target.poll() is None:
            samples.append({"monotonic_s": time.monotonic() - started, "gpus": smi()}); time.sleep(period)
        out, err = target.communicate(timeout=5)
    finally:
        bg_status, bg_out = stop(bg)
    (directory / "stdout.txt").write_text(out if 'out' in locals() else "")
    (directory / "stderr.txt").write_text(err if 'err' in locals() else "")
    (directory / "contender.stdout").write_text(bg_out)
    post = smi(); parsed = parse_log(log)
    pows = [r["power_w"] for s in samples for r in s["gpus"] if r["index"] == 0]
    temps = [r["temperature_c"] for s in samples for r in s["gpus"] if r["index"] == 0]
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "phase": state, "target_gpu": 0, "contender_gpu": CONTENDER_GPU[state], "action": {"skin": skin, "every": every, "delay": 0, "check": "yes"}, "steps": steps, "pre_telemetry": pre, "post_telemetry": post, "during_telemetry": samples, "max_power_w": max(pows, default=None), "max_temperature_c": max(temps, default=None), "power_cap_flag": bool(pows and max(pows) >= 345.0), "lammps": parsed, "success": target.returncode == 0 and "loop_seconds" in parsed, "scientifically_safe": parsed.get("dangerous_builds", 0) == 0, "contender_exit_status": bg_status}
    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--skin", required=True); p.add_argument("--every", type=int, required=True)
    p.add_argument("--sequence", nargs="+", default=["A", "B", "A"], choices=("A", "B", "C", "D")); p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--gpu-build", type=Path, default=ROOT / "build_kokkos_cuda"); p.add_argument("--sample-period", type=float, default=.25)
    p.add_argument("--cool-temp", type=float, default=45); p.add_argument("--cool-power", type=float, default=45); p.add_argument("--cool-timeout", type=float, default=900)
    a = p.parse_args(); out = a.output if a.output.is_absolute() else ROOT / a.output
    if out.exists() and any(out.iterdir()): p.error(f"output is non-empty: {out}")
    lmp = a.gpu_build if a.gpu_build.is_absolute() else ROOT / a.gpu_build
    if lmp.is_dir(): lmp /= "lmp"
    if not lmp.is_file(): p.error(f"missing executable: {lmp}")
    if not target_row(smi()): p.error("GPU 0 is unavailable")
    contender = build_gpu_contender(); out.mkdir(parents=True)
    records = []
    for i, state in enumerate(a.sequence):
        rec = phase(out, f"{i:02d}-{state}", state, a.skin, a.every, a.steps, lmp, contender, a.sample_period, a.cool_temp, a.cool_power, a.cool_timeout)
        records.append(rec); print(json.dumps({"phase": state, "loop": rec["lammps"].get("loop_seconds"), "safe": rec["scientifically_safe"], "power_cap": rec["power_cap_flag"]}), flush=True)
        if not rec["success"] or not rec["scientifically_safe"]: return 1
    (out / "manifest.json").write_text(json.dumps({"sequence": a.sequence, "records": [str(i) for i in range(len(records))]}, indent=2) + "\n")
    return 0
if __name__ == "__main__": raise SystemExit(main())
