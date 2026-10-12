# ampb GPU contention: calibration and reproduction instructions

## Purpose

Measure whether direct contention on the GPUs changes the measured reward for
the existing Shock/NEMD balance action study. The reward is
`r = - action_wall_seconds`, where `action_wall_seconds` includes applying the
selected balance action and running the next 500 MD steps. This document
specifies a controlled state/action measurement; it does not claim that a GPU
RL policy has already been trained.

The prior ampb inventory in `2026-10-08_ampb/README.md` and
`2026-10-09_ampb/PREFLIGHT_STATUS.md` documents a 192-core CPU host but does not
document any allocated NVIDIA GPUs. First verify that the current ampb job
actually has four usable CUDA GPUs. If it does not, stop: CPU contention cannot
answer the GPU-contention question.

## Why ampb contention must be calibrated

On amp2, the KOKKOS/CUDA Shock run used 32 MPI ranks across four RTX 3090 GPUs,
with eight ranks per GPU and CUDA MPS enabled. One CUDA contender ran on each
target GPU with two 128 MiB arrays, 32 arithmetic iterations per element, and
no sleep. In the completed five-repeat matrix, all 240 trials per regime were
safe. Mean 500-step action time was 2.023 s idle and 3.660 s under same-GPU
contention, about 81% slower. That intensity clearly changed the reward on
amp2, but a faster GPU does not automatically need a stronger contender: the
target and contender both run on the same GPU, so their relative scheduling
and kernel mix determine the effect. Choose ampb intensity from measured
slowdown and telemetry, not from its advertised peak performance.

## Required ampb preflight

Run inside the actual GPU allocation and preserve the output in the experiment
directory:

```bash
hostname
date -Is
nvidia-smi -L
nvidia-smi -q
nvidia-smi topo -m
command -v nvcc
nvcc --version
command -v nvidia-cuda-mps-control
mpirun --version
taskset -pc $$
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
```

Proceed only when all of these are true:

- Four NVIDIA GPUs are allocated and usable by this job, with enough free
  memory for the target plus the contender. The current runner uses all four.
- The job has 32 distinct physical CPU cores for MPI ranks and four additional
  available CPU cores for CUDA launch threads. Keep the four contender CPU IDs
  disjoint from the MPI rank cores and choose them near their respective GPUs
  according to `nvidia-smi topo -m` and `lscpu`.
- CUDA, the driver, MPI, and CUDA MPS can initialize. Verify MPS with this
  user's dedicated directories; do not stop a pre-existing MPS server.
- No other LAMMPS/GPU benchmark is running on the allocated GPUs during the
  measurement.

If ampb exposes a scheduler-assigned `CUDA_VISIBLE_DEVICES` mask, retain it.
Set `LAMMPS_GPU_IDS` to the same four device tokens in the same order (for
example, both variables may be `4,5,6,7`). Set
`LAMMPS_GPU_CONTENDER_CPUS` to four available CPU IDs, one per GPU. Do not
replace a scheduler mask with machine-wide GPU IDs.

## Update the checkout and build

Use the current research branch containing the GPU Shock implementation:

```bash
git fetch origin
git switch rl-hpc-research
git pull --ff-only origin rl-hpc-research
git status --short --branch
```

Determine the CUDA compute capability from the allocated GPU model and select
the matching Kokkos architecture name supported by this checkout. For example,
use the Hopper Kokkos target for a Hopper GPU; do not reuse amp2's
`AMPERE86`/`sm_86` setting by default. Then build an independent executable:

```bash
export KOKKOS_ARCH=<Kokkos-architecture-for-this-GPU>
export GPU_CUDA_ARCH=sm_<major><minor>
export LAMMPS_GPU_IDS=<same-four-device-tokens-as-CUDA_VISIBLE_DEVICES>
export LAMMPS_GPU_CONTENDER_CPUS=<four-spare-CPU-IDs-near-the-GPUs>
BUILD_JOBS=16 bash rl_hpc/characterization/shock/build_shock_kokkos_cuda.sh
```

The build script uses `KOKKOS_ARCH` (default `AMPERE86`) and the CUDA contender
builder uses `GPU_CUDA_ARCH` (default `sm_86`). Verify `build_shock_kokkos_cuda/lmp
-h` lists `SHOCK`, `lj/cubic/kk`, `wall/piston/kk`, and `momentum`. Preserve
compiler/MPI/CUDA versions, GPU model and UUIDs, compute capability, MPS
version, build command, binary SHA-256, CPU/GPU affinity, and scheduler
allocation.

## Create ampb-native Shock checkpoints

Use a GPU-generated restart and the fixed eight phase decisions below. Their
decision indices are held constant so the amp2 and ampb studies sample the
same portions of the 60,000-step trajectory; actual ampb telemetry and reward
are measured on ampb. The runner records a 500-step probe under the selected
resource regime before measuring the action interval.

```bash
DATA=rl_hpc/characterization/data/ampb_gpu_contention_v1
python3 rl_hpc/characterization/shock/prepare_gpu_initial.py \
  --output "$DATA/initial-seed47287" --velocity-seed 47287
python3 rl_hpc/characterization/shock/make_restart_spec.py \
  --restart "$DATA/initial-seed47287/checkpoint-S1.restart" \
  --physical-seed 47287 --output "$DATA/initial-restart-list.json"
python3 rl_hpc/characterization/shock/materialize_representative_checkpoints_gpu.py \
  --selection rl_hpc/characterization/shock/configs/gpu_representative_phases_seed47287.json \
  --restart-list "$DATA/initial-restart-list.json" \
  --output "$DATA/representative-checkpoints"
```

Before continuing, confirm 491,520 atoms, all eight checkpoint files and
SHA-256 values, expected step numbers, finite thermodynamic values, zero
dangerous builds during materialization, normal MPI exit, and no residual
LAMMPS/MPI processes. Keep restarts and logs under ignored `characterization/data`;
do not commit them.

## Start MPS and calibrate contention strength

Use unique user-owned MPS directories and set the same paths for LAMMPS and the
contenders:

```bash
export CUDA_VISIBLE_DEVICES="$LAMMPS_GPU_IDS"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/${USER}-lammps-mps-ampb-pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/${USER}-lammps-mps-ampb-log"
rl_hpc/characterization/shock/mps_control.sh status
rl_hpc/characterization/shock/mps_control.sh start
rl_hpc/characterization/shock/mps_control.sh status
```

Run the 60-trial pilot first (two representative phases, skip and factor 1.5,
three regimes, five repetitions). Start with a lower contender intensity than
amp2's medium setting:

```bash
GPU_CUDA_ARCH="$GPU_CUDA_ARCH" python3 -u \
  rl_hpc/characterization/shock/fork_gpu_state_action.py \
  --checkpoints "$DATA/representative-checkpoints" \
  --output "$DATA/pilot-light" --phase pilot --repetitions 5 --seed 20261012 \
  --gpu-buffer-mb 64 --gpu-repeats 8 --gpu-sleep-us 1000
```

For every matching `(phase, action, repetition)`, compare idle and
GPU-contention runtime. The contender must have nonzero CUDA activity on all
four target GPUs throughout the probe and reward interval. Record runtime,
reward, GPU utilization, memory utilization, power, clocks, temperature, MPS
clients, and contender exit status.

Select the lightest intensity that yields a repeatable reward change above
idle measurement noise. As a preregistered practical target, aim for a median
GPU-contention slowdown of 25--100% over idle across the pilot's matched
phase/action cells, with the paired 95% interval for mean runtime difference
above zero. If the pilot is below that target, repeat the pilot at amp2 medium
`(128 MiB, 32 repeats, 0 us)` and only then try heavy `(256 MiB, 64 repeats,
0 us)`. If slowdown is above 2x, power/thermal throttling appears, memory
headroom becomes small, or a contender fails, step down and rerun the pilot.
Change one contender parameter at a time when refining intensity, and save
each setting in a separate output directory. Do not select an intensity from
the full action matrix after inspecting which action wins.

The pilot's purpose is to establish an externally induced reward shift. It
does not establish that the best action changes with contention. Keep those
claims separate.

## Full state/action measurement

After choosing and recording the intensity before the main run, execute the
full matrix once with the same intensity in every GPU-contention trial:

```bash
python3 -u rl_hpc/characterization/shock/fork_gpu_state_action.py \
  --checkpoints "$DATA/representative-checkpoints" \
  --output "$DATA/state-action-main" --phase main --repetitions 5 \
  --seed 20261013 \
  --gpu-buffer-mb <calibrated-MiB> \
  --gpu-repeats <calibrated-repeats> \
  --gpu-sleep-us <calibrated-sleep-us>
```

This is 8 phases × 6 actions (skip and factors 0.5, 0.75, 1.0, 1.25, 1.5) ×
3 regimes (idle, CPU control, direct GPU contention) × 5 repetitions = 720
independent 500-step action measurements. Jobs are run serially in randomized
order. Keep the MPI/GPU mapping, MPS mode, checkpoint, action definition, and
contention intensity fixed. Use the identical fixed intensity for all
GPU-contention cells.

Stop on the first unsafe or nonfinite transition, dangerous neighbor build,
wrong atom/step count, MPI error, missing GPU telemetry, MPS failure, power cap,
or leftover process. Do not resume by retaining partial repetitions in the
analysis. Preserve raw results but exclude an incomplete repetition as a
whole.

## Analysis and report

For each `(phase, action, regime)`, report `n`, mean, sample standard deviation,
CV, and 95% confidence interval for runtime and reward. Pair idle and GPU
contention by `(phase, action, repetition)` and report the mean difference,
paired 95% interval, and percent slowdown. Report the action ranking per phase
and regime and the best-fixed versus state-conditioned descriptive gap. State
that oracle values selected and scored on the same observations are descriptive
only. Compare amp2 and ampb by within-machine percentage changes and action
ranking; absolute seconds alone mix hardware speed with contention effect.

The state/action runner does not train a GPU policy. If the ampb matrix confirms
that contention shifts reward and changes action ranking beyond measurement
noise, record that as evidence for a subsequent GPU-specific policy study. Do
not reuse CPU-trained Q coefficients as if they were GPU-trained.

When all measurements and analysis are complete, stop only the MPS server
started by this experiment:

```bash
rl_hpc/characterization/shock/mps_control.sh status
rl_hpc/characterization/shock/mps_control.sh stop
```

Commit only scripts, compact summaries, and figures. Keep checkpoints, binary,
logs, telemetry JSONL, and raw per-run results out of Git.
