#!/usr/bin/env python3
"""Run the existing official-balance trajectory protocol on KOKKOS/CUDA."""
from __future__ import annotations

import collect_trajectories as collector
import evaluate_official_balance as official
import fork_gpu_state_action as gpu
import shock_counterfactual as base

# Reuse the established protocol and record format; replace only the backend.
base.LMP = gpu.GPU_LMP
collector.Simulation = gpu.GpuSimulation
official.INITIAL_BALANCE = 'balance 1.0 shift x 10 1.0 out initial.mesh'

if __name__ == "__main__":
    official.main()
