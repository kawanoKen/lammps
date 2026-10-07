# Official LAMMPS `fix balance` baseline comparison

This experiment compares four official LAMMPS dynamic-load-balancing
heuristics before making a direct claim against learned policies.

| Label | Configuration |
|---|---|
| A | particle-count `fix balance` |
| B | `fix balance ... weight neigh 1.0` |
| C | `fix balance ... weight neigh 1.5` |
| D | `fix balance ... weight time 1.0` |

All four use `Nfreq=500`, imbalance threshold 1.2, `shift x`, at most 10
iterations, and stop threshold 1.1. The workload, binary restart, initial
factor-1.5 partition, 50-step warm-up, MPI32/OMP1 placement, 120 x 500-step
episode, safety checks, and CPU-contention schedules match the hybrid-policy
evaluation design.

Each heuristic is evaluated four times in idle and four times under piecewise
CPU contention: two held-out physical initial states times two repetitions.
The 32 independent processes are run sequentially, with heuristic order
randomized inside each restart/schedule group. The measured 500-step `run`
wall time includes any internal `fix balance` check, redistribution, migration,
and neighbor reconstruction cost.

A two-restart, four-decision smoke completed for all A--D methods before the
main launch. All eight smoke processes exited normally with zero dangerous
builds. This experiment does not rerun learned policies; a later final paired
comparison may interleave the best official heuristic with both frozen learned
policies.
