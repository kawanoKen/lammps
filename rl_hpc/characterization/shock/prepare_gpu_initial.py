#!/usr/bin/env python3
"""Create a Shock/NEMD initial restart entirely with KOKKOS/CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import shock_counterfactual as base

ROOT = Path(__file__).resolve().parents[3]
LMP = ROOT / "build_shock_kokkos_cuda" / "lmp"


def digest(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--velocity-seed", type=int, default=47287)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gpu_ids = os.environ.get("LAMMPS_GPU_IDS",
                             os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    if len([item for item in gpu_ids.split(",") if item.strip()]) != 4:
        parser.error("LAMMPS_GPU_IDS must name exactly four allocated GPUs")

    params = (base.SOURCE / "shockparams.mod").read_text()
    for name, value in (("nx", 480), ("ny", 16), ("nz", 16)):
        params, count = re.subn(
            rf"(variable\s+{name}\s+index\s+)\d+",
            lambda match, value=value: match.group(1) + str(value), params,
        )
        if count != 1:
            raise RuntimeError(f"failed to set {name}")
    (output / "shockparams.mod").write_text(params)
    setup = (base.SOURCE / "shocksetup.mod").read_text()
    setup, count = re.subn(
        r"(velocity\s+all\s+create\s+\$\{temp\}\s+)87287",
        lambda match: match.group(1) + str(args.velocity_seed), setup,
    )
    if count != 1:
        raise RuntimeError("failed to set velocity seed")
    (output / "shocksetup.mod").write_text(setup)
    (output / "in.baseline").write_text(base.baseline_input(output, (1100,)))

    command = [
        "mpirun", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
        "-np", "32", str(LMP), "-k", "on", "g", "4", "-sf", "kk",
        "-pk", "kokkos", "gpu/aware", "off", "-in", str(output / "in.baseline"),
        "-log", str(output / "log.lammps"), "-screen", str(output / "screen.txt"),
    ]
    env = {
        **os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": gpu_ids,
        "CUDA_MPS_PIPE_DIRECTORY": f"/tmp/{os.environ.get('USER', 'user')}-lammps-mps-pipe",
        "CUDA_MPS_LOG_DIRECTORY": f"/tmp/{os.environ.get('USER', 'user')}-lammps-mps-log",
    }
    started = time.perf_counter()
    result = subprocess.run(command, cwd=output, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, timeout=1800)
    (output / "stdout.txt").write_text(result.stdout)
    (output / "stderr.txt").write_text(result.stderr)
    restart = output / "checkpoint-S1.restart"
    log = (output / "log.lammps").read_text(errors="replace")
    # The bundled baseline intentionally uses ``neigh_modify check no``, so
    # LAMMPS reports "Dangerous builds not checked" during initial generation.
    # Fork measurements switch to check=yes and require an exact zero count.
    safe = (result.returncode == 0 and restart.is_file()
            and "ERROR:" not in log and "Lost atoms" not in log
            and not re.search(r"(?i)(?<![a-z_])(?:nan|[+-]?inf(?:inity)?)(?![a-z_])", log))
    manifest = {
        "version": "shock-gpu-initial-v1", "git_commit": base.git_commit(),
        "lammps": str(LMP), "lammps_sha256": digest(LMP),
        "velocity_seed": args.velocity_seed, "atoms": 491520,
        "mpi_ranks": 32, "omp_threads": 1, "gpu_ids": gpu_ids.split(","), "gpu_count": 4,
        "ranks_per_gpu": 8, "mps": True, "gpu_aware_mpi": False,
        "returncode": result.returncode, "wall_seconds": time.perf_counter() - started,
        "restart": str(restart), "restart_sha256": digest(restart) if restart.is_file() else None,
        "safe": safe, "command": command,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not safe:
        raise RuntimeError("GPU initial trajectory failed safety checks")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
