# Shock/NEMD frozen-policy live evaluation (2026-09-28)

## Completed 138,240-atom experiment

Evaluated four policies on the real CPU/MPI LAMMPS build, MPI 32,
25 decisions × 500 MD steps from the same S1 restart per policy run.
Each policy used a new persistent LAMMPS process; no gradients or online
parameter updates. Five blocks each ran the four policies in randomized order.
All policies started with the same `(skin=0.35, neighbor factor=1.5)`
and the same 50-step warmup plus 50-step initialization, then preserved
the shock state and domain partition across decisions. Every decision applied
one absolute skin/factor and `balance ... weight neigh`, then ran 500 steps.
Only the action+run timer is summed below, not process startup or warmup.

| Policy | 5 total times (s) | Mean ± sample SD (s) | Difference vs fixed (paired mean, 95% t interval) |
| --- | --- | ---: | ---: |
| Fixed `(0.35,1.5)` | 41.924, 41.900, 41.949, 41.916, 41.975 | 41.933 ± 0.029 | reference |
| Observable rule | 40.914, 40.916, 40.944, 40.961, 40.927 | 40.933 ± 0.020 | −1.000 s [−1.043, −0.958] |
| Myopic predictor, seed 22 | 40.994, 40.741, 40.417, 40.501, 40.984 | 40.727 ± 0.267 | −1.205 s [−1.531, −0.880] |
| Sequential FQI, seed 22 | 43.068, 43.237, 43.149, 43.200, 43.087 | 43.148 ± 0.072 | +1.215 s [+1.098, +1.332] |

The predictor improves 2.87% over this fixed setting; the rule improves
2.39%; FQI is 2.90% slower. Predictor minus rule averages −0.205 s,
comparable to predictor's 0.267 s run-to-run SD, so this is **not** evidence
that the predictor beats the simple rule. FQI clearly does not win here.

The rule mostly chooses `(0.5,1.5)` (110/125 decisions); therefore a
constant `(0.5,1.5)` is a necessary stronger fixed comparator. The old
checkpoint-fork best fixed `(0.35,1.5)` was not re-calibrated for persistent,
repeated balancing. Do not interpret the 2.87% contrast as a proven adaptive
benefit over the best possible fixed policy.

All 20 runs completed to step 13,700 with exit code zero and no dangerous
builds. Temperature remained about 127–164; per-atom total energy about
249.319–249.358; shock-position observable about 17.9–156.2.
These are finite/plausible checks, not proof of full physics equivalence.
The single shared physical restart makes the five blocks repeated runtime
measurements, **not five independent shock realizations**. Confidence
intervals therefore characterize repeat noise on this host only.
CVCF/history diagnostics are omitted, as in training. No power/thermal
telemetry was logged; temporal machine drift remains a possible confounder.

Models and split: see [SHOCK_OFFLINE_TRAINING.md](SHOCK_OFFLINE_TRAINING.md).
Raw actions, segment timings and thermo are in the ignored directory
`rl_hpc/characterization/data/shock_live_paired_20260928/`.

## Heavier and longer validation

Following the small-run result, a new 245,760-atom restart was created by
keeping `nx=240` and original thermodynamic/shock parameters while increasing
the transverse dimensions from `ny=nz=12` to 16. The first 50-decision
smoke run (25,000 MD steps; step 1,300→26,300) completed in 171.35 s of
action+run time. It had zero dangerous builds; final front observable 301.07
versus x-box end 427.63; finite temperature 113–163 and per-atom energy
249.309–249.447. Baseline-restart creation took 48.75 s and produced a
66.8 MB restart.

A randomized two-block live comparison of fixed `(0.35,1.5)`, stronger fixed
`(0.5,1.5)`, and the observable rule completed successfully. All six processes
exited zero at step 26,300, with zero dangerous builds:

| Policy | Block 1 / 2 action+run time (s) | Mean (s) |
| --- | ---: | ---: |
| Fixed `(0.35,1.5)` | 176.367 / 176.545 | 176.456 |
| Fixed `(0.5,1.5)` | 171.357 / 171.677 | 171.517 |
| Observable rule | 169.838 / 169.836 | 169.837 |

The rule saved 1.519 and 1.841 s relative to the stronger fixed action
(mean 1.680 s, 0.98%). With only **two** blocks from the same physical seed,
no reliable confidence interval or generalization claim is warranted.
It chose `(0.35,0.8)` for decisions 0–8 and `(0.5,1.5)` thereafter in
*both* runs: it is effectively a reproducible **two-phase schedule**, not
evidence that online observations are necessary. Indeed, its first nine
decisions were **slower** than fixed `(0.5,1.5)` by 2.05 and 1.87 s; the
remaining 41 decisions were faster by 3.57 and 3.71 s. Partition and
simulation histories persist, so a simple per-segment comparison cannot
separate whether this comes from better partition history, trajectory drift,
or machine-state effects. A matched two-phase schedule and multiple physical
restarts are needed before crediting state-conditioned control.

The final front observable was about 300.9–301.1, inside the x-box ending
at 427.63. Temperatures stayed about 113–163; per-atom energies 249.309–249.447.
These are only coarse physical checks.

The 25-decision learned models were **not** applied to 50 decisions or the
larger system: doing so would extrapolate outside their training scope.
Raw data: `rl_hpc/characterization/data/shock_heavy_paired_20260928/`.
