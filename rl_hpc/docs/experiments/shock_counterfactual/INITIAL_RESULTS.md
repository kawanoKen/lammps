# Shock/NEMD: checkpoint-fork counterfactual characterization

> **Correction and scope limitation:** The original `weight time` trials
> execute `run 0` after warm-up, clearing the timing data needed by time
> weighting. They cannot establish the performance of correctly initialized
> time weighting. The conclusion below ruling out state-dependent control
> across all four methods is withdrawn. Balance factor (0.8), iteration count
> (10), and target imbalance (1.0) were fixed, so their numerical control
> opportunities remain untested here. A winner's CV alone does not establish
> significance of an action difference. The fixed/oracle numbers below are
> descriptive, selected on the same data, and not general adaptation bounds.
>
> A new numerical skin × neighbor-weight sweep is now specified in
> [NUMERIC_SWEEP_PROTOCOL.md](NUMERIC_SWEEP_PROTOCOL.md). It tests
> 30 numerical actions at all six checkpoints and independently confirms a
> candidate ranking reversal. Results are isolated from this earlier dataset.

## Question and protocol

This study asks whether the best *absolute* neighbour skin / load-balance
action changes with the internal state of one shock simulation.  It is not an
RL experiment.  A 138,240-atom expansion of
`examples/PACKAGES/shock/nemd/in.nemd` (`nx=240`, `ny=nz=12`) produced one
baseline shock trajectory.  Six binary restart checkpoints were read by new,
independent MPI processes.  Each process applied an action and measured the
following 500 MD steps.

* MPI ranks: 8 and 32, one OpenMP thread per rank; neither oversubscribes the
  48 physical-core host.
* Skin: 0.15, 0.25, 0.35, 0.50, 0.70, 1.00.
* Neighbour policy: `every 1 delay 0 check yes`.
* Balance policy: none; `balance 1.0 shift x 10 1.0`; the same with
  `weight neigh 0.8`; the same with `weight time 0.8`.
* Every action was repeated three times in randomized order.  A common,
  fixed 50-step warm-up precedes each measurement so `weight time` has a
  preceding timing window.  The reported `ACTION_WALL_SECONDS` includes the
  neighbour/balance action application plus the 500-step segment.

The full 6 x 2 x 6 x 4 x 3 matrix contains 864 measurements.  All completed:
all return codes were zero; dangerous-neighbour-build count was zero; no
NaN/Inf or LAMMPS error was recorded.

## Shock states and best actions

`xshock` is the example's shock-position observable at the saved restart.
Values are mean +/- sample standard deviation over three forks.  The
runner-up gap is the gap from the first to second ranked action.

| MPI | State | Step | xshock | Best `(balance, skin)` | Wall s | CV | Runner-up gap |
| ---: | :---: | ---: | ---: | :--- | ---: | ---: | ---: |
| 8 | S1 | 1,100 | 10.85 | neigh, 0.50 | 3.807 +/- 0.015 | 0.38% | 0.57% |
| 8 | S2 | 3,600 | 39.90 | neigh, 0.35 | 3.939 +/- 0.003 | 0.08% | 0.03% |
| 8 | S3 | 6,100 | 68.50 | neigh, 0.35 | 4.387 +/- 0.039 | 0.88% | 1.70% |
| 8 | S4 | 8,600 | 97.65 | neigh, 0.50 | 4.521 +/- 0.015 | 0.33% | 0.83% |
| 8 | S5 | 11,100 | 126.29 | neigh, 0.50 | 4.880 +/- 0.007 | 0.14% | 2.39% |
| 8 | S6 | 13,600 | 154.93 | neigh, 0.50 | 5.149 +/- 0.061 | 1.18% | 1.05% |
| 32 | S1 | 1,100 | 10.85 | neigh, 0.50 | 1.303 +/- 0.001 | 0.07% | 0.76% |
| 32 | S2 | 3,600 | 39.90 | neigh, 0.25 | 1.494 +/- 0.009 | 0.59% | 2.57% |
| 32 | S3 | 6,100 | 68.50 | neigh, 0.50 | 1.547 +/- 0.008 | 0.50% | 4.78% |
| 32 | S4 | 8,600 | 97.65 | neigh, 0.50 | 1.729 +/- 0.011 | 0.62% | 2.61% |
| 32 | S5 | 11,100 | 126.29 | neigh, 0.50 | 1.804 +/- 0.085 | 4.73% | 1.05% |
| 32 | S6 | 13,600 | 154.93 | neigh, 0.50 | 1.588 +/- 0.008 | 0.49% | 2.39% |

## What changes with state?

The application state changes strongly.  With no balancing and skin 0.50,
8-rank `Nlocal_max/Nlocal_mean` grows from 1.33 (S1) to about 1.98 (S3--S6);
32 ranks remain about 2.01.  The 8-rank no-balance wall time rises from 4.62 s
to 11.13 s.  Its Pair time rises from 2.02 s to 3.42 s and Comm time from
0.63 s to 2.42 s.  This is direct evidence of shock-front-driven internal
nonstationarity and increasing imbalance.

The best balance *class* does **not** reverse: `shift x + weight neigh 0.8`
is best at every checkpoint for both MPI counts.  Averaged across all skins
and states it costs 4.70 s (8 ranks) and 1.70 s (32 ranks), versus 10.16 / 3.58
s for no balance.  Atom-count and time weighting are both about 5.61 / 1.93 s.
Thus balancing is valuable, but choosing among the four tested balance modes
is not a state-conditioned control problem in this workload.

Skin has a small state interaction.  At 32 ranks, S2 prefers 0.25 over the
globally preferred 0.50 by 2.57%, well above its 0.59% CV; at 8 ranks, S2's
0.35 vs 0.50 ordering differs by only 0.03% and is not meaningful.  S3 at 8
ranks also prefers 0.35, but the aggregate value remains small.

## Best fixed action versus state oracle

The best fixed action is `(skin=0.50, shift x + weight neigh 0.8)` for both
MPI counts.  Summing checkpoint means:

| MPI ranks | Fixed total, s | State-conditioned oracle, s | Oracle improvement |
| ---: | ---: | ---: | ---: |
| 8 | 26.763 | 26.682 | 0.30% |
| 32 | 9.504 | 9.465 | 0.41% |

Therefore this shock setup is a strong **load-imbalance / balancing-value**
workload, but it does not yet establish a practically useful state-dependent
balance policy.  It provides only weak evidence for state-dependent skin
selection under the tested action grid; the total fixed-policy regret is below
the variability seen in some individual segments (notably 32-rank S5).

## Important implementation scope

The baseline checkpoint trajectory retains the bundled `in.nemd` CVCF/history
instrumentation.  Fork processes restore the physical shock state, NVE and
wall dynamics, and then apply the controlled neighbour/balance settings; the
CVCF/history diagnostic fixes are not reinstantiated in the 500-step forks.
Consequently, conclusions concern the shock NEMD force/communication workload
from the original physical state, not the exact wall time of its post-shock
diagnostic instrumentation.  A follow-up that requires end-to-end example
timing should reinstantiate those diagnostic fixes in every fork.

## Recommendation

Do not use the current balance-mode set as an online RL action space: its
best member is stationary.  The most useful next experiment is a compact
counterfactual study of *rebalance timing/threshold* (for example `fix balance`
frequency and threshold) across the same shock checkpoints.  That tests the
plausible state-dependent decision here: **when** the increasing imbalance
justifies paying a rebalance cost.
