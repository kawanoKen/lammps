# Shock/NEMD hybrid-balance protocol

## Research question

With neighbor skin fixed at 0.50, does the favorable load-balance action vary
with observable shock state or CPU contention?

## Environment

- CPU/MPI LAMMPS, MPI 32, OpenMP 1 thread/rank.
- 491,520 atoms.
- One persistent LAMMPS process per episode.
- 120 decisions per episode and 500 MD steps per decision.
- Idle and controlled piecewise CPU-contention training conditions use equal
  experience budgets.

## State

Policy inputs include temperature, pressure, atom imbalance, current factor,
intervals since balance, previous balance choice/cost, pre/post-balance
imbalance, previous Pair/Neigh/Comm/Modify timing, neighbor-build rate,
rank-local atom/neighbor imbalance, and subdomain-width statistics.

Decision index, MD step, shock position, host load, and the hidden contention
label are excluded. Shock position is retained only for logging and a safety
guard.

## Action and reward

The action is either:

```text
skip balance
```

or:

```text
balance 1.0 shift x 10 1.0 weight neigh FACTOR
FACTOR in [0.5, 1.5]
```

The reward is negative action-application plus 500-step wall time. Skin is
fixed at 0.50 and `neigh_modify every 1 delay 0 check yes` remains fixed.

## Training design

Each condition contains 48 episodes: 20 stratified exploration episodes and
28 online-FQI episodes. Exploration uses exactly 50% skip actions; balance
actions cover 30 equal factor strata twice per episode in randomized order.
The learned greedy action compares skip with a continuous piecewise-quadratic
factor optimum.

## Held-out evaluation

Frozen policies are tested on two physical restart seeds absent from training,
with two repetitions in idle and CPU-jitter regimes. The five policies are:

- idle-trained hybrid policy;
- CPU-jitter-trained hybrid policy;
- always skip;
- always balance at factor 1.5;
- balance at factor 1.5 iff atom imbalance is at least 1.5.

The design has 40 independent 60,000-step runs. Policy order is randomized
within matched restart/schedule groups. Evaluation performs no exploration or
model updates.

## Validity rules

Out-of-range factors, skipped neighbor weighting, dangerous builds,
nonfinite thermo, atom-count changes, unexpected step advances, or MPI errors
invalidate and terminate an episode. Partial episodes are not replayed.
