# LAMMPS RL/HPC experiment preparation

This directory contains the auditable, non-RL scaffolding for later
reinforcement-learning and SMDP experiments.  It does not implement a policy,
agent, replay buffer, or parameter tuner.

The current workload is a research copy of the bundled fixed-size Lennard-Jones
benchmark in `bench/in.lj`: 32,000 atoms, a deterministic velocity seed, and a
configurable number of timesteps.  The input is intentionally kept close to the
upstream benchmark so that the result remains a meaningful LAMMPS computation.

## Build

From the repository root:

```bash
bash rl_hpc/build.sh
./build_rl/lmp -h
```

The current build uses GCC, MPI, OpenMP, the OpenMP package, and the Python
package.  It is a shared-library build so the serial Python proof of concept
can load it directly from `build_rl/` without installing anything system-wide.

The CUDA/KOKKOS build is intentionally independent from `build_rl/`:

```bash
bash rl_hpc/build_kokkos_cuda.sh
./build_kokkos_cuda/lmp -h
```

It targets CUDA 13.4 and the RTX 3090 compute capability (Kokkos
`AMPERE86`), with MPI, KOKKOS, CUDA, and Serial backends enabled.  It does not
modify or reuse the CPU build directory.

## Run one experiment

Each output directory must be new; this avoids silently overwriting a run.

```bash
python3 rl_hpc/run_experiment.py \
  --mpi 4 \
  --steps 1000 \
  --seed 87287 \
  --output rl_hpc/runs/example-mpi4
```

The launcher records `result.json`, the exact command in `command.txt`,
`stdout.txt`, `stderr.txt`, `screen.out`, and the LAMMPS `log.lammps`.  Extra
input variables can be supplied without editing the input file:

```bash
python3 rl_hpc/run_experiment.py \
  --mpi 4 --steps 1000 \
  --lammps-var skin=0.4 \
  --lammps-var neigh_every=10 \
  --output rl_hpc/runs/neighbor-example
```

`--omp-threads` sets `OMP_NUM_THREADS` and is recorded.  The default is one
thread per MPI rank, which avoids accidental oversubscription in the baseline.

## Interactive intervention smoke test

```bash
python3 rl_hpc/interactive_smoke.py
```

This loads the locally built shared library through the source-tree Python
module, runs two short segments in one simulation, reads the current step and
temperature, and changes one neighbor-list scheduling setting between the
segments.  It is serial by design; a future multi-rank library wrapper should
use a matching `mpi4py`/MPI installation.

## Files and results

- `bench_lj.in`: adjustable 32,000-atom LJ input derived from `bench/in.lj`.
- `build.sh`: reproducible CMake configuration and build command.
- `run_experiment.py`: repeatable process launcher and JSON result writer.
- `interactive_smoke.py`: minimal segmented-run Python API demonstration.
- `build_kokkos_cuda.sh`: reproducible independent CUDA/KOKKOS build.
- `kokkos_cuda_smoke.py`: short 1-GPU/4-GPU and GPU-aware-MPI smoke matrix.
- `ENVIRONMENT.md`: detected machine, toolchain, repository, and build facts.
- `PARAMETERS.md`: candidate runtime controls separated by safety level.
- `BASELINE.md`: repeated-run timing sanity check on this machine.
- `CHARACTERIZATION.md`: measured state/action matrix, contention episodes, and
  research conclusions; no RL algorithm is implemented.
- `CONTENTION_CHARACTERIZATION.md`: GPU factorial/network-contestation status,
  mechanisms, and blocked-resource accounting.
- `NETWORK_ENVIRONMENT.md`: multi-node/network capability findings and limits.
- `build_characterization_cpu.sh`: separate CPU build for the SPC/E/PPPM probe.
- `build_kokkos_cuda_characterization.sh`: separate CUDA build including
  KSPACE/MOLECULE/RIGID for the SPC/E GPU follow-up.
- `runs/`: generated experiment outputs; ignored by git.

## CUDA/KOKKOS smoke test

Run the small GPU validation after building the independent executable:

```bash
python3 rl_hpc/kokkos_cuda_smoke.py \
  --steps 100 \
  --output "rl_hpc/runs/kokkos-cuda-smoke-$(date +%Y%m%d-%H%M%S)"
```

The script records every command, stdout/stderr, LAMMPS log, wall time, and
`result.json`.  It runs a CPU reference, one GPU and four GPU cases, both
`gpu/aware off` and `gpu/aware on`, and compares the one-GPU `thermo` values
against the CPU reference.  On this host the `off` cases succeed and the
`gpu4_aware_on` probe returns 139 because Open MPI was built without
CUDA-aware support; this is classified explicitly as an expected diagnostic
result, not treated as a working configuration.  Use `gpu/aware off` for
current multi-GPU runs.

## Non-stationarity characterization

The characterization uses a separate CPU build with KSPACE/MOLECULE/RIGID so
that both required workloads are available. It does not modify `build_rl/`:

```bash
bash rl_hpc/build_characterization_cpu.sh
python3 rl_hpc/characterization/characterize.py \
  --output "rl_hpc/characterization/data/matrix-$(date +%Y%m%d-%H%M%S)" \
  --matrix-reps 3 --stability-reps 5
python3 rl_hpc/characterization/analyze_characterization.py \
  rl_hpc/characterization/data/<new-run>/measurements.jsonl \
  --episodes rl_hpc/characterization/data/episodes-final2/episodes.jsonl \
  --internal-episodes rl_hpc/characterization/data/episodes-idle-final/episodes.jsonl
```

The selected workloads are a 128,000-atom Lennard-Jones system and a 36,000-
atom SPC/E/PPPM system. The runner measures idle, CPU-contended, memory-
contended, and same-GPU-contended states with a small neighbor-list action
set. The segmented probe is:

```bash
python3 rl_hpc/characterization/segmented_episode.py \
  --output "rl_hpc/characterization/data/episode-$(date +%Y%m%d-%H%M%S)" --segment-steps 20
python3 rl_hpc/characterization/segmented_episode.py \
  --output "rl_hpc/characterization/data/idle-$(date +%Y%m%d-%H%M%S)" --segment-steps 20 --idle-only
```

The full state/action tables and conclusions are in `CHARACTERIZATION.md`.
Raw logs and JSONL under `characterization/data/` are ignored because they can
become large.

The next GPU mechanism experiment is a separated 3x3 factorial with explicit
same-GPU/different-GPU/CPU-only controls:

```bash
python3 rl_hpc/characterization/gpu_factorial.py \
  --output "rl_hpc/characterization/data/gpu-factorial-$(date +%Y%m%d-%H%M%S)" \
  --workload lj --gpu-build build_kokkos_cuda --steps 500 --reps 5
```

`CONTENTION_CHARACTERIZATION.md` records the current result and any runtime
blockers. `NETWORK_ENVIRONMENT.md` documents why the network matrix must wait
for a real multi-node allocation; `network_runner.py` requires an explicit
allocated host list.

## Current limitations

The baseline executable remains CPU/MPI/OpenMP capable and is kept in
`build_rl/`.  The separate `build_kokkos_cuda/` executable is a working
CUDA/KOKKOS build for the four RTX 3090 GPUs.  This is a smoke-tested setup,
not yet a tuned multi-GPU benchmark: the host reports no NVLink/P2P-read path
between the cards, and the installed Open MPI is not CUDA-aware.

The machine currently uses the `powersave` CPU governor.  Timing results are
therefore machine-state measurements, not a claim of peak hardware
performance.  Use fresh output directories and keep the process count and
thread count fixed when comparing configurations.

## Recommended next research step

Use the segmented Python/library pattern as the interface boundary: define a
small validated action space around neighbor-list scheduling, MPI/OpenMP
placement, and prevalidated KOKKOS launch choices, collect observations from
LAMMPS timing data after each segment, and only then add an offline dataset
schema.  Do not allow an agent to change physical timestep or force-field
parameters in that first environment.
