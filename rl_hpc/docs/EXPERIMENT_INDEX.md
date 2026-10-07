# Experiment index and current evidence

This is the entry point for completed LAMMPS characterization and control
experiments. A training reward curve is not treated as policy evidence;
checkpoint-fork comparisons and held-out live evaluations take precedence.

## Current result map

Date- and machine-specific result bundles begin with
[2026-10-07 — amp2](experiments/2026-10-07_amp2/README.md).

| Experiment | Controlled action | Regime/state | Favored policy or action | Strength of conclusion | Documents |
|---|---|---|---|---|---|
| GPU LJ contention | neighbor skin and `every` | idle vs same-GPU contention | idle: about `(0.6,5)`; same-GPU load: about `(1.0,20)` | Positive state-action interaction; workload-specific | [results](experiments/gpu_lj_contention/RESULTS.md) |
| Initial internal-state controls | neighbor settings | evolving LJ and SPC/E checkpoints | nearly stationary best setting | Negative/control result | [results](experiments/internal_state/RESULTS.md) |
| Shock initial counterfactual | skin and four balance classes | six shock checkpoints, MPI 8/32 | `shift x + weight neigh 0.8` generally favored | Balance valuable, but original `weight time` comparison is invalid and numerical ranges were narrow | [initial results](experiments/shock_counterfactual/INITIAL_RESULTS.md) |
| Shock numerical sweep | skin × neighbor-weight factor | six shock checkpoints, MPI 32 | best action changes by checkpoint | Independently confirmed action reversal; descriptive fixed/oracle gap 2.64% | [protocol](experiments/shock_counterfactual/NUMERIC_SWEEP_PROTOCOL.md), [results](experiments/shock_counterfactual/NUMERIC_SWEEP_RESULTS.md) |
| Shock full-action online FQI | discrete skin × factor | idle and CPU jitter | idle-trained policy won both tested regimes | Beats tested fixed settings, but progress-like features confound interpretation | [protocol](experiments/shock_full_action/PROTOCOL.md), [results](experiments/shock_full_action/RESULTS.md) |
| Shock continuous skin only | continuous skin; factor fixed | idle and CPU jitter | fixed skin 0.50 | Learned control did not improve runtime | [implementation/training](experiments/shock_skin_only/IMPLEMENTATION_AND_TRAINING.md), [results](experiments/shock_skin_only/RESULTS.md) |
| Shock hybrid balance | skip or continuous neighbor-weight factor; skin fixed | idle and CPU jitter | idle: idle-trained; jitter: jitter-trained in mean | Both beat tested fixed/threshold baselines. Idle-side difference is resolved; jitter-side policy reversal remains inconclusive | [protocol](experiments/shock_balance_hybrid/PROTOCOL.md), [implementation](experiments/shock_balance_hybrid/IMPLEMENTATION.md), [results](experiments/shock_balance_hybrid/RESULTS.md) |
| Shock representative-state forks | skip or neighbor-weight factor; skin fixed | eight states from one fixed-initial-state idle trajectory | best action changes from skip to factors 0.75--1.50 | 240 safe forks show a descriptive fixed/oracle gap of 3.45%; independent persistence validation is pending | [results](experiments/shock_balance_hybrid/REPRESENTATIVE_STATE_ACTION_RESULTS.md), [portable reproduction](experiments/shock_balance_hybrid/PORTABLE_REPRODUCTION.md) |

## Current interpretation

- Shock/NEMD creates substantial endogenous MPI load imbalance. Rebalancing is
  necessary in the tested long trajectory.
- Dynamic skin control alone is not useful under the tested formulation.
- Continuous neighbor-weight factor selection is the strongest current CPU
  control candidate. Skip decisions were rare in learned policies.
- Same-GPU contention changes the preferred neighbor policy for GPU LJ, but
  that result is separate from the CPU shock action space.
- No current result proves that RL is superior to the best possible fixed
  factor, a strong supervised contextual optimizer, or a better phase rule.
- The highest-priority confirmation is independent reproduction of the strong
  representative-state action reversals and measurement of whether their
  partition effects persist beyond the first 500-step interval.

## Non-experiment references

- [Problem formulation](research/PROBLEM_FORMULATION.tex)
- [Source-code adaptation analysis](research/SOURCE_CODE_ANALYSIS.md)
- [Candidate parameter space](research/PARAMETER_SPACE.md)
- [Machine environment](environment/MACHINE.md)
- [Network environment](environment/NETWORK.md)
- [Baseline timing](environment/BASELINE.md)
