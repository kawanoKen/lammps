# Shock trajectory offline training — 2026-09-28

## Scope and outcome

Executed training, not just framework preparation: six checkpoints were saved
(three myopic models and three sequential FQI models). No live policy evaluation
or online parameter update was performed. These results do **not** demonstrate
that RL speeds up LAMMPS.

The shock CPU workload uses 32 MPI ranks, 500 MD steps per decision, 25 decisions
per episode, absolute actions: skin {0.25,0.35,0.50,0.70,1.00} × neighbor weight
{0.50,0.65,0.80,1.00,1.20,1.50}. Every action applies
`balance 1.0 shift x 10 1.0 weight neigh FACTOR`; `every 1 delay 0 check yes`.
There is no skip-balance action, no delta action, and no time-weight action.
CVCF/history diagnostics are omitted consistently with the fork experiment.

## Dataset and audit

575 transitions from 23 persistent episodes: pilot 75 (epsilon 0.4) and additional
500 (epsilon 0.8). The provisional rule chooses (0.5,1.5) if observed atom
imbalance exceeds 1.2, otherwise (0.35,0.8); exploration is uniform over 30 actions.
The rule is a collection heuristic, not a demonstrated optimal policy.

Validated all transitions for 500-step advance, exact next-state/current-state
continuity, reward = negative measured action-plus-run seconds, timing-component
sum, behavior propensity, safe flag, and zero dangerous builds. Input features
are finite. This is not a full scientific-equivalence validation.

Whole-episode split: 15 train / 4 validation / 4 test (375/100/100 transitions).
Split is deterministic and stratified by collection batch. All episodes share
one physical restart; this tests unseen behavior trajectories, not unseen physics.
Five phases are defined by decision indices 0–4, 5–9, 10–14, 15–19, 20–24.

| Data | Actions visited per phase (out of 30) | Missing phase/action cells |
| --- | --- | ---: |
| All | 30, 27, 30, 30, 29 | 4 |
| Train | 27, 24, 26, 26, 28 | 19 |

Global action coverage is complete, but state-conditioned coverage is not.
Even a visited phase/action cell need not have sufficient replication, and phases
do not capture the full partition/history state.

## Models and reproducibility

Implementation: `rl_hpc/characterization/shock/train_trajectories.py`.
Uses existing NumPy; no new dependencies or build changes.

Features use only pre-action data: physical observables, current skin/factor,
remaining episode fraction, previous per-step timing/build counts, rank atom /
ghost / neighbor statistics, and 32 domain widths extracted from the stored mesh.
Normalizing previous metrics per step accounts for the initial 50-step observation
window versus later 500-step windows. State summaries may still be partially
observed; atom positions and neighbor-list history are not fully represented.

Train-only mean/std normalization. Shared ridge model includes quadratic action
terms and state/action interactions. Ridge is selected from {1,10,100} using
validation one-step prediction MSE, then shared with sequential FQI.
Seeds 11/22/33 bootstrap whole training episodes (not neural initialization seeds).

* Myopic: gamma=0, one regression; also the supervised runtime predictor. These
  are the same mathematical baseline, not two independent methods.
* Sequential: gamma=0.97, 25 fitted-Q iterations, max over all 30 actions.
  This study explicitly treats decision 25 as a finite-horizon terminal boundary,
  despite collection records calling it truncation. Remaining time is a feature.
  Bootstrap continuation targets are clipped to [minimum training reward ×
  remaining decisions, 0]. This numerical guard is not a certified return bound
  or a conservative offline-RL guarantee. Out-of-support max-Q bias is possible.

The saved `.npz` files contain weights, normalization, and action ordering.
The manifest identifies the source-file hashes, script hash, commit, split,
features, action grid and seeds; metrics preserve training losses and choices.
No hyperparameters/checkpoints were selected using test errors below.

Reproduce to a **new** directory (existing output is refused):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 \
  rl_hpc/characterization/shock/train_trajectories.py \
  --output rl_hpc/characterization/data/shock_training_reproduction
```

## Offline results

| Seed | Ridge | Predictor validation RMSE (s) | Test MAE (s) | Test RMSE (s) |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 100 | 0.1148 | 0.1042 | 0.1267 |
| 22 | 10 | 0.1140 | 0.0945 | 0.1221 |
| 33 | 10 | 0.1194 | 0.1072 | 0.1316 |

These errors concern logged actions only, not the quality of selected actions.
The predicted-action runtime is not an observed speedup.

Sequential FQI has material warning signs. Logged test-action Q ranges are
[-18.557,+0.062], [-18.872,-1.711], and [-17.200,-2.037] for seeds 11/22/33.
Positive Q is inconsistent with strictly negative rewards and reflects regression
error. Seed 33 chooses (1.0,1.5) for 51/100 test states; other seeds choose much
more dispersed actions. Do not interpret the finite losses or different action
choices as proof of a valid sequential improvement. No estimated policy return
or oracle recovery is reported, since this dataset alone does not establish it.

## Artifacts and next step

Generated artifacts (ignored): `rl_hpc/characterization/data/shock_training_v1/`
contains `manifest.json`, `metrics.json`, and six `.npz` checkpoints.

**Next research step:** run a small paired live evaluation of these frozen models
against the collection rule and a calibrated fixed numerical action, from identical
initial states. Report total elapsed time and failures rather than predicted Q;
treat the sequential models as experimental, not deployment-ready. The held-out
logged errors cannot resolve whether these policies actually improve runtime.
