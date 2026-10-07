#!/usr/bin/env bash
# Run smoke first, then the authorized three-episode pilot; stop on failure.
set -euo pipefail
cd /work/kawano/lammps
out="$1"
mkdir -p "$out"
exec >>"$out/workflow.log" 2>&1
trap 'rc=$?; printf "%s\n" "$rc" >"$out/exit_code"; date -Is >"$out/finished_at"' EXIT
date -Is >"$out/started_at"
python3 -u rl_hpc/characterization/shock/collect_trajectories.py --output "$out/smoke" --episodes 1 --decisions 2
python3 -u rl_hpc/characterization/shock/collect_trajectories.py --output "$out/pilot" --episodes 3 --decisions 25 --epsilon 0.4
