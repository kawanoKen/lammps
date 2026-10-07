#!/usr/bin/env bash
# Independent smoke, then 8 long MPI32 exploratory episodes. Stop on first failure.
set -euo pipefail
cd /work/kawano/lammps
out="$1"
restart=/work/kawano/lammps/rl_hpc/characterization/data/shock_heavy_baseline_20260928/checkpoint-S1.restart
mkdir -p "$out"
exec >>"$out/workflow.log" 2>&1
trap 'rc=$?; printf "%s\n" "$rc" >"$out/exit_code"; date -Is >"$out/finished_at"' EXIT
date -Is >"$out/started_at"
python3 -u rl_hpc/characterization/shock/collect_trajectories.py \
  --output "$out/smoke" --restart "$restart" --expected-atoms 245760 \
  --episodes 1 --decisions 2 --epsilon 0.8 --seed-base 20261100
python3 -u rl_hpc/characterization/shock/collect_trajectories.py \
  --output "$out/dataset" --restart "$restart" --expected-atoms 245760 \
  --episodes 8 --decisions 50 --epsilon 0.8 --seed-base 20261101
