# Shock/NEMD hybrid balance-control study

Status: idle and CPU-jitter training started sequentially on 2026-10-05 at
14:35 JST as `lammps-shock-balance-hybrid-v1.service`. Idle training is
running first. The
study fixes neighbour skin at 0.50 and asks the controller both whether to
rebalance and, when it does, which continuous neighbor-weight factor to use.

## Formulation

- Action: `skip balance`, or execute
  `balance 1.0 shift x 10 1.0 weight neigh FACTOR`, where
  `FACTOR` is a real value in `[0.5,1.5]`. Skin remains 0.50 and
  `neigh_modify every 1 delay 0 check yes` remains fixed.
- State: temperature, pressure, atom imbalance, current factor, intervals
  since the last balance, previous balance choice/cost and imbalance before
  and after it, previous Pair/Neigh/Comm/Modify times and neighbor-build
  rate, rank nlocal/neighbor imbalance, and domain-width statistics.
  Decision index, MD step, shock position, host load, and the hidden
  contention label are excluded from policy features. Shock position is
  retained only for logging and the front-position safety guard.
- Reward: negative wall seconds for the balance decision plus the following
  500 MD steps. Thus skip avoids direct balance cost, while any downstream
  change in Pair/Comm/Neigh time remains in the reward.
- Experience per condition: **48 episodes**, comprising 20 pure exploration
  episodes and 28 online-FQI episodes, each with 120 decisions. This is more
  exploration than the previous 12+24 studies. The idle and CPU-jitter
  conditions use identical budgets, physical restart order, seed and learner
  settings. Estimated total collection time is about 26--30 hours.
- Exploration: exactly 50% skip actions. The balance half covers 30 equal
  factor strata twice per episode in randomized order. Later epsilon
  exploration uses the same construction. Greedy factor selection maximizes
  the learned piecewise-quadratic response over the continuous range and
  compares it directly against skip. FQI backup uses a 41-point factor grid;
  executed greedy factors are not restricted to that grid.
- Safety: factor values outside `[0.5,1.5]`, skipped neighbor weighting,
  dangerous builds, nonfinite/invalid thermo, atom-count change, unexpected
  step advance, or MPI failure abort the episode. Partial episodes are never
  replayed. Each episode is a separate persistent LAMMPS process internally
  segmented into 120 decisions.

Separate two-episode/four-decision smokes completed in idle and CPU-jitter
modes without error. They validate mechanics only, not performance.

## Resilience and outputs

The enabled user service uses `Restart=on-failure`, a 30-second delay, and
at most three rapid restart attempts per hour. User linger is enabled. On
restart, completed episodes are reconstructed into replay and an incomplete
episode is preserved before being rerun. After 48+48 episodes, a later start
exits without recollecting data.

```bash
systemctl --user status lammps-shock-balance-hybrid-v1.service
journalctl --user -u lammps-shock-balance-hybrid-v1.service -f
```

Generated data remain ignored under:

```text
rl_hpc/characterization/data/shock_balance_hybrid_idle_v1/
rl_hpc/characterization/data/shock_balance_hybrid_cpu_jitter_v1/
```

Training traces will not establish policy value. After both policies freeze,
the required test is paired live evaluation on the two held-out physical
restarts under identical idle and CPU-contention schedules. Baselines must
include always-skip, always-balance with a training-selected fixed factor,
and the previous full-action policy. The key questions are whether the
learned balance frequency is state-dependent and whether CPU contention
changes the preferred skip/factor decision by more than measurement noise.

## Held-out evaluation

Both 48-episode training collections completed on 2026-10-06: 5,760 valid
transitions per regime and 11,520 total.  All 96 processes exited normally;
no dangerous build, unsafe transition, NaN, or LAMMPS error was found.

The paired frozen-policy evaluator is
`evaluate_balance_hybrid.py`.  It compares the idle-trained and
CPU-jitter-trained policies with always-skip, always-balance at factor 1.5,
and an atom-imbalance threshold rule.  Factor 1.5 was selected before this
test from the existing numerical sweep restricted to skin 0.50.  The fixed
threshold rule balances at factor 1.5 iff atom imbalance is at least 1.5.
It is not tuned on held-out results.

The main design has two held-out physical restarts, two repeats, idle and
piecewise CPU contention, and five policies: 40 independent 60,000-step
processes.  All methods receive the same restart and contention schedule in
each paired group.  Policy order is randomized.  Frozen learned policies
perform no exploration or model update.  A 10-process, four-decision smoke
completed and passed the independent result auditor before the main launch.

Generated output is ignored under
`rl_hpc/characterization/data/shock_balance_hybrid_eval_main_v1/`.

### Completed held-out results

The main evaluation completed all 40/40 independent processes on 2026-10-07.
Every run contained 120 valid 500-step decisions, exited normally, and had
zero dangerous builds, nonfinite values, or recorded LAMMPS errors. Values
below are mean +/- sample standard deviation over four paired runs per cell
(two held-out physical seeds and two schedule repetitions).

| Regime | Idle-trained | CPU-jitter-trained | Always skip | Fixed 1.5 | Threshold 1.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Idle | **788.29 +/- 4.15 s** | 814.88 +/- 5.73 s | 2120.85 +/- 5.57 s | 932.16 +/- 5.84 s | 972.52 +/- 9.41 s |
| CPU jitter | 920.18 +/- 5.18 s | **909.58 +/- 11.25 s** | 2587.67 +/- 9.91 s | 988.25 +/- 6.94 s | 1035.29 +/- 10.80 s |

Relative to fixed factor 1.5, idle training saved 143.86 s (15.43%) in
idle and 68.06 s (6.89%) under jitter. Jitter training saved 117.28 s
(12.58%) in idle and 78.67 s (7.96%) under jitter. Each of the four paired
runs favored both learned policies over fixed 1.5 in each regime.

The learned-policy ordering changes by regime. In idle, idle training beat
jitter training by 26.58 s (3.37%); the paired 95% t interval for
`jitter - idle` was [13.66, 39.51] s. Under jitter, jitter training was
10.60 s (1.15%) faster on average, but the paired differences changed sign
across the four runs and their 95% interval was [-36.45, 15.24] s. Thus the
mean ordering reverses, but only the idle-side advantage is currently
resolved beyond sampling variability. The data do not yet prove that CPU
contention requires a distinct policy.

Always-skip was extremely poor, showing that redistribution is essential in
this growing shock imbalance. The threshold rule was also slower than always
balancing at factor 1.5. Learned policies balanced in roughly 113--116 of 120
intervals, so their advantage is mainly continuous factor selection rather
than frequent skip decisions. The idle-trained policy used a mean executed
factor near 0.97; the jitter-trained policy used about 1.24. Both traversed
the full approved [0.5, 1.5] range.

These results establish superiority over the tested fixed and threshold
baselines, not over the globally best possible fixed factor, a stronger
supervised contextual optimizer, or an oracle. There are only two independent
held-out physical initial states. The next decisive test is therefore a
larger paired confirmation focused on the two frozen learned policies under
CPU jitter, plus a dense fixed-factor baseline selected without using those
confirmation outcomes.
