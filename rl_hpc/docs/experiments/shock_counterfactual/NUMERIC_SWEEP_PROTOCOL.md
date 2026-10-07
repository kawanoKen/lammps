# Numerical skin × neighbor-weight sweep

This experiment tests state-dependent numerical action ordering, not a choice
between balance algorithms. It reuses six original physical restart states,
32 MPI ranks, a common 50-step warm-up, and 500 measured MD steps. All forks
restore NVE/walls and consistently omit CVCF/history diagnostics. No new
physical trajectory or RL policy is trained.

Actions are the full Cartesian product of skins 0.25, 0.35, 0.5, 0.7, 1.0 and
neighbor-weight factors 0.5, 0.65, 0.8, 1.0, 1.2, 1.5. The balance command is
`balance 1.0 shift x 10 1.0 weight neigh FACTOR`. Neighbor settings remain
`every 1 delay 0 check yes`. Preparation builds the selected neighbor list
before balancing. This is deliberate for neighbor weighting; time weighting
is not tested because this preparation would clear its timer history.

Twenty-four endpoint checks cover both skin/factor extremes at all six states.
Discovery uses five randomized repetition blocks (900 trials). A strongest
candidate opposite ordering is selected using discovery data and frozen in
`confirmation_plan.json`. Ten fresh repetitions of each of its two actions
at both states test that ordering independently (40 trials). Bootstrap
intervals use these fresh repetitions only. No claim of reversal is based on
one candidate's CV or the discovery minimum alone.

The exploratory fixed/oracle gap is (fixed-oracle)/fixed, with equal weights
for checkpoints. It is selection-biased and not an episode-policy result.
Results are conditional on one trajectory; boundary optima do not establish
a global optimum and warrant a subsequent explicit range extension.

Run from the repository root:

```bash
python3 rl_hpc/characterization/shock/numeric_sweep.py --output rl_hpc/characterization/data/shock_numeric_v1 --phase preflight
python3 rl_hpc/characterization/shock/numeric_workflow.py --output rl_hpc/characterization/data/shock_numeric_v1
```

The same command resumes missing trials; completed rows are never rerun.
Incomplete attempts are preserved under separate attempt directories. Locks
prevent concurrent writers. A failed safety check stops collection. Each MPI
job has a 180-second timeout with process-group cleanup. Discovery has a
seven-hour bound. Manifests hash checkpoints, the binary, and sweep code.

Outputs: `measurements.jsonl`, `matrix.csv`, `response.svg`, `REPORT.md`, and
`workflow_complete.json`, under the ignored output directory. Timers separately
record preparation, balance, and run time; their sum is checked against total
action wall time during preflight inspection. The process-startup time is
recorded but not used to rank actions.
