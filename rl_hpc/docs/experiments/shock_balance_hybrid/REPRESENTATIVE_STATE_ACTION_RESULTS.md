# Representative-state balance action forks

## Protocol

One fixed physical initial state was run three times with the official
`fix balance 500 1.2 shift x 10 1.1 weight neigh 1.0` heuristic.  Eight
representative physical states were selected from the reproducible trajectory.
At each state, the MPI partition was reconstructed with `weight neigh 1.0`
outside the timer because a LAMMPS binary restart does not preserve the
nonuniform processor boundaries.  Independent MPI32 processes then evaluated
six actions for 500 MD steps: skip balance, or execute one `balance` command
with neighbour-weight factor 0.50, 0.75, 1.00, 1.25, or 1.50.  Each action was
repeated five times in randomized order.  The reward interval includes action
application and the subsequent 500-step run.

All 240 trials completed with zero dangerous builds, crashes, step mismatches,
or non-finite thermo values.

## Immediate runtime results

| Phase | Step | Best action | Mean, s | Std, s | CV | Gap to runner-up |
|---:|---:|:---|---:|---:|---:|---:|
| 1 | 4,800 | skip | 4.6608 | 0.0163 | 0.35% | 2.36% |
| 2 | 15,300 | factor 0.75 | 6.0743 | 0.0419 | 0.69% | 0.18% |
| 3 | 20,300 | factor 1.50 | 5.9008 | 0.0171 | 0.29% | 6.29% |
| 4 | 27,300 | factor 1.25 | 6.2871 | 0.0896 | 1.43% | 2.23% |
| 5 | 32,300 | factor 1.25 | 6.9175 | 0.0397 | 0.57% | 1.54% |
| 6 | 42,300 | factor 1.00 | 7.5156 | 0.0528 | 0.70% | 0.94% |
| 7 | 48,800 | factor 0.75 | 6.7554 | 0.0889 | 1.32% | 4.55% |
| 8 | 57,800 | factor 0.75 | 6.5513 | 0.0614 | 0.94% | 2.70% |

Mean runtime matrix (seconds):

| Phase | skip | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.6608 | 6.5226 | 5.3900 | 4.7709 | 4.9510 | 5.0666 |
| 2 | 6.0854 | 7.0222 | 6.0743 | 6.0886 | 6.1821 | 6.3729 |
| 3 | 6.2718 | 6.8123 | 6.6584 | 6.6622 | 6.8594 | 5.9008 |
| 4 | 6.4447 | 7.1542 | 6.5678 | 6.4275 | 6.2871 | 6.6204 |
| 5 | 7.0508 | 7.2323 | 7.0932 | 7.0240 | 6.9175 | 7.2659 |
| 6 | 7.5949 | 7.8247 | 7.5860 | 7.5156 | 7.7530 | 8.1919 |
| 7 | 7.0970 | 7.0626 | 6.7554 | 7.0661 | 7.4934 | 8.2478 |
| 8 | 7.2698 | 6.7281 | 6.5513 | 7.2961 | 8.0354 | 8.8239 |

The best fixed action over the eight equally weighted representative states is
skip, with a summed mean of 52.4753 s.  Selecting the measured best action at
each state gives 50.6628 s, a descriptive improvement of 3.45%.  This value is
selected and evaluated on the same samples; it is not yet an unbiased policy
improvement estimate.

## Interpretation

The immediate best action changes substantially along one physical trajectory.
Several gaps are larger than within-action CV, especially phases 1, 3, 4, 7,
and 8.  Phase 2 is inconclusive because its 0.18% ranking gap is below the
observed variability.  Actions also alter the resulting rank decomposition:
for example, at phase 8 the mean post-segment atom imbalance ranges from 1.916
at factor 0.50 to 3.994 at factor 1.50, while neighbour imbalance ranges from
about 1.04 to 1.41.  Thus action affects both reward and the next software
state, although persistence beyond one 500-step interval has not yet been
measured.

The next required validation is a targeted independent rerun of the winner and
runner-up actions at the strongest reversal phases, followed by a common-action
continuation for 2,000--5,000 steps.  That will test ranking reproducibility and
whether the altered partition has sequential value rather than only an
immediate contextual effect.
