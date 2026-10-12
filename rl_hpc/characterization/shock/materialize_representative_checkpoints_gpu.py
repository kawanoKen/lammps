#!/usr/bin/env python3
"""Materialize representative Shock checkpoints using KOKKOS/CUDA."""
from __future__ import annotations

import collect_trajectories as collector
import fork_gpu_state_action as gpu
import materialize_representative_checkpoints as materialize
import shock_counterfactual as base
import evaluate_official_balance as official

base.LMP = gpu.GPU_LMP
collector.Simulation = gpu.GpuSimulation
official.INITIAL_BALANCE = 'balance 1.0 shift x 10 1.0 out initial.mesh'
materialize.POLICY = 'official_time_10'

if __name__ == "__main__":
    materialize.main()
