# Frozen shock-policy cross-regime evaluation

The live evaluation **completed successfully** on 2026-10-04 at 01:45 JST
as the user service `lammps-shock-online-cross-v1.service`, in
`rl_hpc/characterization/data/shock_online_cross_main_v1/`.
The initial interactive launch stopped when its session ended after about
11 decisions, before completing any episode. The incomplete directory was
preserved with an `-interrupted-` suffix and is excluded from comparisons.
The service resumed the first episode on 2026-10-03 at 17:28 JST.
This is **held-out, paired cross-regime testing**, not K-fold retraining.
The idle-trained and CPU-jitter-trained FQI checkpoints are frozen: no
exploration and no model updates occur during evaluation.

## Design

- Physical restarts: seeds 47287 and 57287, absent from either training set.
- Regimes: no deliberate contention; controlled piecewise CPU contention on
  physical CPUs 0--7, alternating every 2--5 decision intervals.
- Policies: idle-trained; CPU-jitter-trained; fixed `(skin=0.5, factor=1.5)`;
  fixed `(skin=0.35, factor=1.5)`. Both fixed settings are plausible strong
  baselines, not deliberately poor controls.
- Two repetitions per physical seed and regime. Each repetition uses one
  recorded contention schedule shared by all four policies; execution order
  is randomized within each matched group. This yields 32 independent
  LAMMPS processes when complete.
- MPI 32, 491,520 atoms, 120 decisions x 500 steps; same initial partition,
  100 setup steps, reward timer and physics validity guards as training.
  The measured metric is the sum of action-application plus 500-step times;
  startup and the initial 100 steps are outside it.
- The data manifest records restart hashes, model hashes, binary hash,
  schedules, and execution order. Each transition logs action, runtime,
  physics observables, contention state and dangerous-build count.

The primary comparison is paired episode wall-time difference on the same
restart and schedule, not comparison of training curves. Two restarts x two
repetitions per regime give only four pairs per regime: confidence intervals
will be wide and conclusions must reflect that. Repeating the same physical
restart does not create an independent scientific initial condition.

## Reproduction and status

```bash
python3 rl_hpc/characterization/shock/evaluate_online_cross.py \
  --output rl_hpc/characterization/data/shock_online_cross_main_v1 \
  --restart-list rl_hpc/characterization/data/shock_online_seed_restarts_v1/test_restart_list.json \
  --decisions 120 --repeats 2 --regimes idle jitter \
  --policies idle_trained jitter_trained fixed_05_15 fixed_035_15
```

After an interruption, use the same arguments with `--resume` to continue
without deleting completed runs. The incomplete attempt is preserved.

```bash
systemctl --user status lammps-shock-online-cross-v1.service
journalctl --user -u lammps-shock-online-cross-v1.service -f
```

```bash
python3 rl_hpc/characterization/shock/analyze_online_cross.py \
  rl_hpc/characterization/data/shock_online_cross_main_v1
```

The paired results are audited below. GPU shock/NEMD is a separate build/physics
validation task: the current CUDA/KOKKOS binary lacks the input's `lj/cubic`
pair style, so these CPU policies cannot simply be rerun with `-sf kk`.

## Completed results

The analysis audited **32/32 runs**, each with 120 complete 500-step
decisions, zero dangerous-neighbour builds and normal MPI exit. Times below
are measured action-application plus MD-run time per 60,000-step episode,
excluding process startup and the common setup/warm-up. Values are mean +/-
sample standard deviation over four runs (two physical restarts, two
repetitions each). The two repetitions of one restart are not independent
physical initial conditions.

| Regime | Idle-trained | CPU-jitter-trained | Fixed 0.50/1.50 | Fixed 0.35/1.50 |
| --- | ---: | ---: | ---: | ---: |
| Idle | 790.34 +/- 3.05 s | 834.38 +/- 5.12 s | 931.97 +/- 3.64 s | 977.20 +/- 3.96 s |
| CPU jitter | 914.14 +/- 5.22 s | 934.39 +/- 2.91 s | 985.92 +/- 6.64 s | 1021.39 +/- 3.48 s |

Relative to the stronger of the two tested fixed settings (0.50/1.50), the
idle-trained policy saved **141.63 s (15.20%) in idle** and **71.78 s
(7.28%) with CPU jitter**. All four matched pairs had positive savings in
both regimes. The CPU-jitter-trained policy also beat this fixed setting,
but was slower than the idle-trained policy by 44.04 s in idle and 20.26 s
under jitter on average. The idle-trained policy won this head-to-head on
each of the four matched pairs in both regimes.

The frozen idle-trained policy chose skin 0.50 for about 78% of decisions
and 0.70 for the remainder, with factors mostly 1.20 early and 0.50 late.
Its action frequencies were nearly identical across idle and jitter runs.
The CPU-jitter-trained policy chose skin 0.50 for about 77% and 0.35 for
the remainder, with factor 1.50 on about 73% of decisions. Neither policy
used skip-balance in live evaluation. This pattern is compatible with a
**shock-phase schedule**, not evidence that the policies detect and react
to the CPU load regime. The direct balance command averaged about 0.008 s
per decision; total runtime differences were chiefly in the subsequent
MD-run time. This does not prove redistribution has no downstream cost.

Final temperature, energy per atom, and shock position were finite and
similar across policies (mean final temperature about 147.6 and shock
position about 648.7--649.0). That is a coarse physics sanity check, not
a trajectory-equivalence proof.

**Scope:** This is held-out restart testing with only two independent
physical seeds, not K-fold retraining. The fixed baseline covers two
plausible settings, not the globally best fixed action over all 35 actions.
Consequently the measured gain over these fixed baselines is real for the
tested episodes, but does not yet establish an advantage of RL over a
well-tuned fixed configuration, a phase schedule, or a supervised predictor.
It also does not support a claim that CPU contention requires a different
action: the idle-trained policy was best in both regimes. The most useful
next control is a frozen, non-learning phase schedule constructed from the
idle-policy trace, evaluated on the same held-out restarts and schedules.
