# Offline RL status

This document records the first implementation gate; no offline policy result
is claimed yet.

## Implemented formulation

`offline_rl/env.py` is a persistent GPU-0 LAMMPS environment with a fixed
500-step decision interval, reward `-wall_seconds`, delta actions
`(delta_skin, delta_every)`, clipping, requested/applied action logging, and
pre-decision GPU telemetry. It never exposes a contention label in the
observation. Its conservative provisional safe region is
`skin=[0.6,1.2]`, `every=[1,15]`, `delay=0`, `check=yes`; this excludes the
known dangerous low-skin/high-every area until a proper preflight is complete.

`build_kokkos_cuda_rl/` is a newly built shared CUDA/KOKKOS/Python library;
the existing CPU and executable-only CUDA builds were not changed.

## Validation result and blocker

Using the shared GPU library, one process successfully executed `reset()` and
then `step((+0.1,+5))` in the same LAMMPS simulation. The resulting absolute
configuration was `(skin=0.7,every=10)`, the applied action was logged, the
500-step reward was `-1.10709 s`, and no validity failure occurred.

At Python interpreter shutdown, Kokkos CUDA aborts in stream teardown with
`cudaErrorCudartUnloading`. This is a CUDA/Kokkos shared-library teardown
ordering issue, not a failed segment. It is now isolated by
`offline_rl/episode_worker.py`: each whole episode runs in a dedicated worker,
flushes every JSONL transition, and exits without CPython unloading the CUDA
library. `offline_rl/validate_worker.py` ran three persistent 500-step
transitions and verified worker exit 0, unsafe transitions 0, and no residual
GPU compute processes through `nvidia-smi`.

## Dataset instrumentation and retained partial collection

The collector now has signal-safe process-group cleanup: interruption first
terminates the episode worker, whose handler terminates its external CUDA
co-runner. A deliberately interrupted collection left no GPU compute process.
The 33 complete pre-instrumentation episodes (330 transitions) were recovered
as `characterization/data/offline-rl-main-500/transitions.partial.jsonl`.
They cover all five hidden regimes and all five `delta_every` branches,
contain zero unsafe transitions, and have no contention-label leakage into the
observation. Their mean measured 500-step segment time is 1.6392 s (standard
deviation 0.5173 s). This is lifecycle/coverage evidence only, not the final
offline-learning dataset: it predates per-segment LAMMPS timing capture.

Floating-point arithmetic initially made the coverage script report spurious
clipping. Tolerance-aware comparison confirms that the conservative behavior
policy applied no real clipping in the retained data.

The environment now parses the final LAMMPS loop of every segment and records
the timing breakdown, performance, neighbor-build count, and dangerous-build
count in the observable *previous-segment* state. A fresh two-decision
idle/heavy GPU worker validation recorded Pair/Neigh/Comm timing, 25 neighbor
builds, zero dangerous builds, two safe transitions, and no residual GPU
process. New training data will therefore meet the required application
telemetry and safety-record format.

The fresh-schema pilot found a real unsafe point at `(skin=0.67208,every=15)`
under idle conditions: LAMMPS reported one dangerous neighbor build despite
finite thermo quantities. The collector stopped after flushing that transition
and intentionally did not create a merged dataset. Accordingly the safe
rectangle is narrowed to `skin=[0.6,1.2]`, `every=[1,10]`; it preserves all
five delta branches while excluding the observed high-`every` unsafe region.
The reduced boundary is rechecked under idle and heavy same-GPU contention
before final data collection resumes.

That boundary recheck completed six independent runs:
`skin={0.6,0.8,1.2}`, `every=10`, crossed with idle and heavy same-GPU
contention. All six had zero dangerous builds and finite thermo values. The
final-schema pilot (`offline-rl-pilot-v4`) then completed 5 whole episodes / 50
transitions with zero unsafe transitions, zero dangerous builds, zero actual
action clipping, zero power-cap flags, complete LAMMPS timing fields in all
records, and no ground-truth label in the state. Its 500-step mean runtime was
1.4139 s (standard deviation 0.4375 s); the small pilot is a schema/safety
gate, not a performance comparison. It authorizes the 500-transition main
collection using the same independent-process and episode-start cooldown
protocol.

## Safety preflight and contention-worker validation

The representative configurations `(skin,every)` = `(0.6,1)`, `(0.6,5)`,
`(1.0,1)`, and `(1.0,5)` completed under both idle and same-GPU heavy
contention at 500 steps: eight of eight runs succeeded with zero dangerous
builds. The current pilot domain remains conservative rather than claiming
that every point in the original `[0.4,1.2] x [1,20]` proposal is safe.

The worker now supports an external same-GPU contender. A three-transition
heavy worker smoke test completed with no unsafe transition and no residual
GPU compute process. Its stored observation contains only current
configuration, prior-segment data, and pre-decision `nvidia-smi` telemetry;
the `contention_ground_truth` field is separate analysis metadata.

## Piecewise-regime collector smoke test

`offline_rl/collect_dataset.py` now generates whole-episode splits. It keeps
each sampled regime for 2--5 decisions, switches only at 500-step boundaries,
and writes per-episode JSONL plus a merged transition file. A two-episode,
12-transition smoke pilot completed successfully: runtime range 0.816--2.280
s, zero unsafe transitions, no GPU residual process, and no ground-truth
regime key in `state_before`. The smoke sample covered idle, medium-low,
medium-high, and heavy; it is a lifecycle/format validation only, not a
sufficient training dataset.

## 100-transition attempt

A 10-episode / 100-transition collection was started with the strict 45 C /
45 W episode-start baseline. It was stopped because the resulting cooldown
latency made the run impractically long in the current interactive execution
budget. Seven partial episode files were flushed before interruption. The
interruption exposed one orphaned light CUDA contender; it was identified by
PID, terminated, and a subsequent `nvidia-smi` check showed no compute
process. The partial directory is diagnostic only and is not a dataset.

Before resuming collection, `collect_dataset.py` needs signal-safe worker
process-group cleanup. The experimental protocol should retain the strict
baseline for evaluation episodes, but use a documented less-strict cooldown
or batched episode allocation for pilot collection so the required 500--1000
transitions are practical.
