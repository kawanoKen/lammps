#!/usr/bin/env python3
"""Minimal segmented-run demonstration using the LAMMPS Python library."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BUILD_DIR = REPO_ROOT / "build_rl"


def run_segments(lammps_class, segment_steps: int) -> int:
    lmp = lammps_class(cmdargs=["-log", "none", "-screen", "none"])
    try:
        lmp.commands_list(
            [
                "variable seed index 87287",
                "units lj",
                "atom_style atomic",
                "lattice fcc 0.8442",
                "region box block 0 20 0 20 0 20",
                "create_box 1 box",
                "create_atoms 1 box",
                "mass 1 1.0",
                "velocity all create 1.44 ${seed} loop geom",
                "pair_style lj/cut 2.5",
                "pair_coeff 1 1 1.0 1.0 2.5",
                "neighbor 0.3 bin",
                "neigh_modify delay 0 every 20 check no",
                "fix integrator all nve",
                "thermo 5",
                "thermo_style custom step temp etotal",
            ]
        )
        lmp.command(f"run {segment_steps}")
        first = {
            "step": lmp.get_thermo("step"),
            "temperature": lmp.get_thermo("temp"),
        }

        # This is an execution-only intervention: it changes when neighbor
        # lists are rebuilt, not the force cutoff or physical timestep.
        lmp.command("neigh_modify delay 0 every 10 check no")
        lmp.command(f"run {segment_steps}")
        second = {
            "step": lmp.get_thermo("step"),
            "temperature": lmp.get_thermo("temp"),
        }
        print(json.dumps({"first_segment": first, "second_segment": second}, indent=2))
    finally:
        lmp.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-steps", type=int, default=5)
    args = parser.parse_args()
    if args.segment_steps < 1:
        raise SystemExit("--segment-steps must be positive")

    library = BUILD_DIR / "liblammps.so"
    if not library.exists():
        raise SystemExit(f"shared library not found: {library}; run rl_hpc/build.sh first")

    # The source-tree wrapper expects the shared object beside the package.
    # Build a temporary package copy with a symlink to the local library.  This
    # keeps the source tree and all system/user site-packages untouched.
    with tempfile.TemporaryDirectory(prefix="lammps-python-") as temp_dir:
        package_dir = Path(temp_dir) / "lammps"
        shutil.copytree(REPO_ROOT / "python" / "lammps", package_dir)
        (package_dir / "liblammps.so").symlink_to(library)
        sys.path.insert(0, temp_dir)
        from lammps import lammps  # pylint: disable=import-outside-toplevel

        return run_segments(lammps, args.segment_steps)


if __name__ == "__main__":
    raise SystemExit(main())
