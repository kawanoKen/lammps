#!/usr/bin/env python3
"""Materialize imbalance-selected checkpoints on the KOKKOS/CUDA trajectory."""
from __future__ import annotations

import collect_trajectories as collector
import evaluate_official_balance as official
import fork_gpu_state_action as gpu
import materialize_representative_checkpoints as materialize
import shock_counterfactual as base

base.LMP = gpu.GPU_LMP
collector.Simulation = gpu.GpuSimulation
official.INITIAL_BALANCE = 'balance 1.0 shift x 10 1.0 out initial.mesh'
materialize.POLICY = 'no_balance'

if __name__ == "__main__":
    materialize.main()
