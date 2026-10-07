# Larger, longer shock/NEMD trajectory collection (2026-09-28)

## Purpose and protocol

This is a **new exploration dataset**, not an evaluation of the previously
trained small-system policy. The problem is 245,760 atoms (`nx=240`,
`ny=nz=16`) versus 138,240 atoms (`nx=240`, `ny=nz=12`) before; the original
initial density, temperature, timestep, impact speed and axial length are
unchanged. The MPI rank count is 32, with one thread per rank. Each episode
begins from the same step-1,100 binary restart, then keeps one LAMMPS process
alive for 50 decisions × 500 MD steps, ending at step 26,300. Domain
decomposition and physical state carry across decisions; the process is
fully terminated between episodes.

The absolute action grid is skin {0.25,0.35,0.50,0.70,1.00} × `weight neigh`
factor {0.50,0.65,0.80,1.00,1.20,1.50}. `neigh_modify every 1 delay 0 check yes`
and the balance command shape remain fixed. The provisional behavior rule
selects `(0.50,1.50)` when preceding local-atom imbalance exceeds 1.2,
otherwise `(0.35,0.80)`. With probability 0.8, exploration instead samples
uniformly among the 30 actions; exact behavior probability is stored per row.
The rule is not a learned or proven optimal policy.

`reward = -T`, where `T` measures each action's neighbor preparation,
rebalance and following 500-step run. Process startup, the common 50-step
warmup, initial balance and 50-step first-observation run are outside reward.
This matches the earlier trajectory collector. Every action performs a
rebalance; skip-balance and rebalance timing are **not** explored here.
The CVCF/history diagnostics from the original example are omitted in the
fork episodes, as in the smaller trajectory dataset.

## Execution and audit

The expanded restart was created successfully in 48.75 s and occupies
66,847,949 bytes. A separate 50-decision fixed-action safety run completed
before collection: front-position observable 301.07 versus x-box end 427.63,
no dangerous builds. The collection itself first completed a 2-decision smoke
episode, then all eight planned episodes. The background workflow exited 0;
no residual target LAMMPS process was observed afterward.

| Measure | Result |
| --- | ---: |
| Episodes / decisions | 8 / 50 each |
| Transitions | 400 |
| Explored decisions | 317 (79.25%) |
| Unique actions | 30 / 30 |
| Action visit range | 6–80 |
| Sum of episode process wall times | 1,313.4 s |
| Episode process wall-time range | 159.95–168.75 s |
| Episode action+run time range | 156.91–165.79 s |
| Dangerous builds / crashes | 0 / 0 |

The independent audit validated, for every transition: 500-step progression,
exact next-state/current-state continuity, atom count, positive finite
thermodynamic values, action membership, behavior propensity, reward/timer
identity, and zero dangerous builds. Temperature spans 113.22–163.33;
per-atom total energy 249.309–249.439; front-position observable 19.13–301.06.
These coarse checks do not establish scientific equivalence of every action
trajectory, and the shock positions should be interpreted as the example's
observable, not a fully independent front tracker.

**Coverage warning:** in ten 5-decision phases, distinct actions visited were
20, 23, 20, 20, 21, 20, 18, 19, 20, 17 out of 30. Thus 102 of 300
phase/action cells are unvisited. Global action coverage alone is not enough
for reliable state-conditioned action comparison. All eight episodes share
one physical restart; initial action and behavior-policy seed vary, but
different initial thermodynamic realizations are not represented. Do not
randomly split transitions into train/test or claim workload generalization.

## Artifacts and reproduction

Generated files are ignored by Git:

* `rl_hpc/characterization/data/shock_heavy_baseline_20260928/` — expanded
  baseline input/restart, command and execution result.
* `rl_hpc/characterization/data/shock_heavy_trajectory_20260928/` — smoke,
  workflow log/exit code, episode JSONL, manifest, summary and `audit.json`.

Code: `prepare_heavy.py`, `collect_trajectories.py`,
`heavy_trajectory_background.sh`, `audit_trajectories.py` under
`rl_hpc/characterization/shock/`.

To audit the existing dataset again:

```bash
python3 rl_hpc/characterization/shock/audit_trajectories.py \
  rl_hpc/characterization/data/shock_heavy_trajectory_20260928/dataset
```

The original dataset and fixed-policy evaluation are reported separately in
[archived offline training](../../archive/SHOCK_OFFLINE_TRAINING.md) and
[archived live evaluation](../../archive/SHOCK_LIVE_EVALUATION.md). No model was retrained
on these new trajectories.
