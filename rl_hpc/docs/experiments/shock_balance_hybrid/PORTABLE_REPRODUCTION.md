# Portable reproduction: shock state × balance action

This protocol reproduces the fixed-initial-state MPI32 experiment on another
server. It intentionally uses only scripts committed to the repository. Raw
restarts, logs, and measurements are generated under
`rl_hpc/characterization/data/` and remain ignored by Git.

## Scope and comparability requirements

The target workload is Shock/NEMD with `nx=480`, `ny=nz=16` (491,520 atoms),
one OpenMP thread per MPI rank, and 32 MPI ranks without oversubscription. The
physical velocity seed is fixed at 47287. Do not silently reduce the rank count
or workload dimensions: that creates a different experiment. If the server has
fewer than 32 available physical CPU cores, record the limitation and stop.

The experiment has no external CPU/GPU contention. Avoid concurrent benchmark
jobs. Record CPU model, NUMA topology, memory, OS, compiler, CMake, MPI version,
and Git commit before building.

## 1. Dependencies and build

Required commands are CMake, a C++ compiler, Open MPI-compatible `mpirun`, and
Python 3. Python packages are listed in
`rl_hpc/characterization/shock/requirements_state_action.txt`.

```bash
python3 -m pip install -r rl_hpc/characterization/shock/requirements_state_action.txt
BUILD_JOBS=16 bash rl_hpc/characterization/shock/build_shock_cpu.sh
build_shock_char/lmp -h
```

The dedicated build enables MPI, OpenMP, and the SHOCK package and leaves other
existing build directories untouched.

## 2. Generate one fixed physical initial state

```bash
python3 rl_hpc/characterization/shock/prepare_heavy.py \
  --nx 480 --ny 16 --nz 16 --velocity-seed 47287 \
  --output rl_hpc/characterization/data/portable_shock_initial_v1

python3 rl_hpc/characterization/shock/make_restart_spec.py \
  --restart rl_hpc/characterization/data/portable_shock_initial_v1/checkpoint-S1.restart \
  --physical-seed 47287 \
  --output rl_hpc/characterization/data/portable_shock_initial_v1/restart_list.json
```

Require a zero return code, 491,520 atoms, and an existing restart before
continuing.

## 3. Smoke-test state collection

```bash
python3 rl_hpc/characterization/shock/evaluate_official_balance.py \
  --output rl_hpc/characterization/data/portable_state_smoke_v1 \
  --restart-list rl_hpc/characterization/data/portable_shock_initial_v1/restart_list.json \
  --decisions 2 --repeats 1 --regimes idle \
  --policies official_neigh_10 --expected-atoms 491520
```

Inspect `result.json`, `status.json`, and both transition rows. Stop on any
dangerous build, non-finite value, wrong atom count, nonzero exit status, or
step mismatch.

## 4. Collect the frequently visited state trajectory

```bash
python3 rl_hpc/characterization/shock/evaluate_official_balance.py \
  --output rl_hpc/characterization/data/portable_state_trajectory_v1 \
  --restart-list rl_hpc/characterization/data/portable_shock_initial_v1/restart_list.json \
  --decisions 120 --repeats 3 --regimes idle \
  --policies official_neigh_10 --expected-atoms 491520
```

This is the official heuristic
`fix balance 500 1.2 shift x 10 1.1 weight neigh 1.0`. The three independent
processes begin from the same physical state. They estimate execution noise;
they are not different physical initial states.

## 5. Select and materialize eight representative states

```bash
python3 rl_hpc/characterization/shock/select_representative_states.py \
  --input rl_hpc/characterization/data/portable_state_trajectory_v1 \
  --count 8 \
  --output rl_hpc/characterization/data/portable_state_trajectory_v1/representative_states.json

python3 rl_hpc/characterization/shock/materialize_representative_checkpoints.py \
  --selection rl_hpc/characterization/data/portable_state_trajectory_v1/representative_states.json \
  --restart-list rl_hpc/characterization/data/portable_shock_initial_v1/restart_list.json \
  --output rl_hpc/characterization/data/portable_representative_checkpoints_v1 \
  --expected-atoms 491520
```

Require eight checkpoint files, eight SHA-256 digests, complete status, and
zero dangerous builds.

## 6. Smoke-test and collect the state × action matrix

Every fork first reconstructs the same pre-action MPI partition with
`weight neigh 1.0` outside the reward timer. This is necessary because a binary
restart preserves the physical state but not the prior nonuniform processor
boundaries. The measured actions are skip balance and factors 0.50, 0.75, 1.00,
1.25, and 1.50 with skin fixed at 0.5.

```bash
python3 rl_hpc/characterization/shock/fork_representative_balance_actions.py \
  --checkpoints rl_hpc/characterization/data/portable_representative_checkpoints_v1 \
  --output rl_hpc/characterization/data/portable_balance_forks_smoke_v1 \
  --phase smoke --repetitions 5

python3 rl_hpc/characterization/shock/fork_representative_balance_actions.py \
  --checkpoints rl_hpc/characterization/data/portable_representative_checkpoints_v1 \
  --output rl_hpc/characterization/data/portable_balance_forks_main_v1 \
  --phase main --repetitions 5
```

The main matrix is 8 states × 6 actions × 5 repetitions = 240 independent MPI
processes. Action order is randomized within each repetition block. Do not run
multiple trials concurrently.

## 7. Analyze without changing the protocol

```bash
python3 rl_hpc/characterization/shock/analyze_representative_balance_actions.py \
  --input rl_hpc/characterization/data/portable_balance_forks_main_v1 \
  --output-json rl_hpc/characterization/data/portable_balance_forks_main_v1/analysis.json \
  --output-md rl_hpc/characterization/data/portable_balance_forks_main_v1/ANALYSIS.md
```

Report the complete runtime matrix, per-action mean/std/CV, best action per
state, best fixed action, and the descriptive state-oracle gap. A ranking change
smaller than its measurement variability is inconclusive. The state oracle is
selected and evaluated on the same data and is not an unbiased policy result.

## Failure policy

Do not modify source or experimental scripts to work around a failure. Preserve
all logs and report the exact command, return code, and error. Do not use
oversubscription, change the action grid, add contention, or continue after an
unsafe trial. Generated data must not be committed.
