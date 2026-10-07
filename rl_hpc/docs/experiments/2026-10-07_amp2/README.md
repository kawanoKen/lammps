# 2026-10-07 — amp2 Shock/NEMD state-action characterization

## Identity and question

- Date: 2026-10-07 (Asia/Tokyo)
- Machine: `amp2.g2.gsic.titech.ac.jp` (`amp2`)
- Repository branch: `rl-hpc-research`
- LAMMPS base commit: `c8bd2ae5927ee236a8892dbd51c18a92cc9c33cf`
- Result commits: `3be966cff1`, `20a1377d5f`

This experiment asks two separate questions from the same physical shock state:

1. Does the selected balance action change the next state distribution?
2. Does that transition effect persist long enough to change the best
   finite-horizon action?

This is not a policy-learning experiment. It is a checkpoint-fork causal
characterization of `T(s,a)` and the action-conditioned transition.

## Environment and workload

The experiment ran on amp2 with 2 × Intel Xeon Gold 6240R CPUs, 48 physical
cores, 768 GB RAM, GCC 15.2, Open MPI 5.0.10, and a dedicated MPI/SHOCK LAMMPS
build at `build_shock_char/lmp`. Each trial used 32 MPI ranks, one OpenMP thread
per rank, and no oversubscription or deliberate external contention.

The workload is an enlarged bundled `examples/PACKAGES/shock/nemd/in.nemd`:

- lattice dimensions: `nx=480`, `ny=nz=16`
- atoms: 491,520
- velocity seed: 47287
- fixed physical initial restart
- skin: 0.5
- neighbor policy: `every 1 delay 0 check yes`
- one decision interval: 500 MD steps

## Frequently visited state collection

The official heuristic

```text
fix balance 500 1.2 shift x 10 1.1 weight neigh 1.0
```

was run from the same initial state in three independent processes. Each
episode contained 120 × 500-step intervals. Total episode times were 831.998,
832.665, and 833.174 s, giving CV 0.071%. All 360 transitions were safe.

The physical trajectory was effectively identical across repetitions until a
small late partition bifurcation. Eight representative states were chosen from
observable application/partition features, excluding step and shock position
from the selection distance.

| Phase | Step | Atom imbalance | Neighbor imbalance | Pair time/step |
|---:|---:|---:|---:|---:|
| 1 | 4,800 | 1.112 | 1.374 | 0.005210 |
| 2 | 15,300 | 1.459 | 1.349 | 0.006627 |
| 3 | 20,300 | 1.639 | 1.305 | 0.007234 |
| 4 | 27,300 | 1.878 | 1.228 | 0.008120 |
| 5 | 32,300 | 2.051 | 1.251 | 0.008794 |
| 6 | 42,300 | 2.099 | 1.061 | 0.009389 |
| 7 | 48,800 | 3.123 | 1.128 | 0.008161 |
| 8 | 57,800 | 3.537 | 1.158 | 0.006775 |

Here atom imbalance is maximum rank atom count divided by the mean. Neighbor
imbalance is defined analogously for neighbor count. High atom imbalance is not
automatically poor balance: dense shock regions can require more neighbor work,
so a neighbor-weighted partition intentionally assigns them fewer atoms.

## Immediate state × action matrix

At each representative physical checkpoint, independent MPI processes tested:

- skip balance
- `weight neigh` factor 0.50, 0.75, 1.00, 1.25, or 1.50

Each action was repeated five times in randomized order, for 240 trials. All
trials completed with zero dangerous builds, crashes, non-finite thermo values,
or step mismatches.

A binary LAMMPS restart does not preserve nonuniform MPI domain boundaries.
Consequently every fork first reconstructed the same factor-1.0 partition
outside the reward timer. The measured interval then included the tested action
and the following 500 steps.

| Phase | Best action | Mean, s | Std, s | CV | Runner-up gap |
|---:|:---|---:|---:|---:|---:|
| 1 | skip | 4.6608 | 0.0163 | 0.35% | 2.36% |
| 2 | factor 0.75 | 6.0743 | 0.0419 | 0.69% | 0.18% |
| 3 | factor 1.50 | 5.9008 | 0.0171 | 0.29% | 6.29% |
| 4 | factor 1.25 | 6.2871 | 0.0896 | 1.43% | 2.23% |
| 5 | factor 1.25 | 6.9175 | 0.0397 | 0.57% | 1.54% |
| 6 | factor 1.00 | 7.5156 | 0.0528 | 0.70% | 0.94% |
| 7 | factor 0.75 | 6.7554 | 0.0889 | 1.32% | 4.55% |
| 8 | factor 0.75 | 6.5513 | 0.0614 | 0.94% | 2.70% |

Phase 2 is inconclusive because its ranking gap is below observed variability.
Phases 1, 3, 4, 7, and 8 provide stronger candidate reversals.

The best fixed action over the eight equally weighted states was skip, with a
summed mean of 52.4753 s. Selecting the measured winner at every state gave
50.6628 s, a descriptive 3.45% improvement. This oracle is selected and
evaluated on the same samples and is not an unbiased policy result.

## Does action change the next state?

Yes, for the software/execution part of the state. It does not measurably alter
the physical trajectory over one interval.

Across actions from the same checkpoint:

- temperature, pressure, and density after 500 steps were identical;
- rank-local atom, ghost, and neighbor distributions changed;
- Pair, Neigh, and Comm timing changed;
- repeated executions of the same action produced the same partition
  statistics, while timing retained ordinary wall-clock noise.

For example, at phase 8 the resulting atom imbalance ranged from 1.916 at
factor 0.50 to 3.994 at factor 1.50. Neighbor imbalance ranged from about 1.04
to 1.41. Across the eight phases, the action-induced Comm-time spread was about
3--19 times the pooled within-action standard deviation; the Pair-time spread
was about 4--16 times that deviation.

Thus the empirical transition is approximately:

```text
physical next state: nearly action-independent
software/partition next state: strongly action-dependent
```

Equivalently, the current evidence supports

```text
P(x_physical,next | s,a) ≈ P(x_physical,next | s)
P(x_software,next | s,a1) != P(x_software,next | s,a2)
```

This conclusion excludes the trivial fact that the state records the selected
configuration itself; the rank partition and timing signals also differ.

## Persistence and finite-horizon reversal

The candidate winner and runner-up at phases 1, 3, 4, 7, and 8 were rerun in
seven randomized independent pairs. The tested action was applied for the first
500 steps. Both branches then used the same no-balance continuation for four
more 500-step intervals. All 70 trials were safe.

Differences are first action minus second action; negative is faster.

| Phase | Comparison | Immediate difference, s (95% paired CI) | 2,500-step difference, s (95% paired CI) |
|---:|:---|---:|---:|
| 1 | skip − factor 1.0 | -0.0940 [-0.1121, -0.0758] | -0.4968 [-0.7739, -0.2196] |
| 3 | factor 1.5 − skip | -0.3595 [-0.3893, -0.3297] | -6.4568 [-6.6274, -6.2862] |
| 4 | factor 1.25 − factor 1.0 | -0.2274 [-0.3366, -0.1183] | +1.5315 [+1.3859, +1.6771] |
| 7 | factor 0.75 − factor 0.5 | -0.2585 [-0.3483, -0.1687] | -1.3467 [-1.6651, -1.0283] |
| 8 | factor 0.75 − factor 0.5 | -0.2565 [-0.3572, -0.1557] | -0.1743 [-0.6565, +0.3080] |

Phase 4 is the key sequential result. Factor 1.25 is faster over the immediate
500-step action interval, but factor 1.0 is faster over the common-continuation
2,500-step horizon. Phase 3 shows a strong persistent effect whose advantage
grows in later segments. Phase 8 has a resolved immediate difference but an
inconclusive cumulative ordering.

## Interpretation

The amp2 results establish all of the following for the tested trajectory:

1. Shock evolution creates substantial endogenous load-state change.
2. The preferred balance action changes with that state.
3. Action changes the next software/partition state beyond simply changing the
   recorded configuration field.
4. The transition effect can persist for several decision intervals.
5. Immediate and finite-horizon action rankings can reverse.

This supports an MDP/sequential-control formulation more strongly than a pure
contextual-bandit formulation. It does not prove that reinforcement learning is
better than model-predictive control, a learned transition/runtime model, or a
carefully designed phase-aware controller.

## Limitations

- One fixed physical initial state and one machine were used.
- There was no deliberate external contention.
- Representative states came from one official heuristic's occupancy
  distribution; other policies may visit additional states.
- The eight-state oracle is descriptive and in-sample.
- Only selected action pairs received the 2,500-step persistence test.
- Cross-machine reproducibility remains to be measured.

## Artifacts

Generated data are ignored by Git:

- `rl_hpc/characterization/data/shock_official_state_trajectory_fixed47287_v1/`
- `rl_hpc/characterization/data/shock_representative_checkpoints_fixed47287_v1/`
- `rl_hpc/characterization/data/shock_representative_balance_forks_main_v1/`
- `rl_hpc/characterization/data/shock_balance_persistence_main_v1/`

Related committed documents:

- [Detailed state-action results](../shock_balance_hybrid/REPRESENTATIVE_STATE_ACTION_RESULTS.md)
- [Portable cross-machine reproduction protocol](../shock_balance_hybrid/PORTABLE_REPRODUCTION.md)
- [amp2 machine record](../../environment/MACHINE.md)
