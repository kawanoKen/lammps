# LAMMPS non-stationarity characterization

Generated 2026-09-21T18:17:39.571837+00:00 from `rl_hpc/characterization/data/20260922-final/measurements.jsonl`.

This is a characterization study, not an RL implementation. The analysis uses LAMMPS loop time, not process startup wall time, as the primary performance measure.

## Executive result

The CPU co-runner produces a clear application state change, especially for the PPPM workload, but the tested CPU neighbor action is nearly state-invariant. The GPU workload is different: same-GPU contention changes the best neighbor configuration, and the fixed-policy penalty is large enough to be practically relevant.

## Measurement stability

| Workload | Backend | Action | Mean loop (s) | Std (s) | CV | n |
|---|---|---|---:|---:|---:|---:|
| lj | cpu | lj_skin03_every1 | 1.7562 | 0.0058 | 0.33% | 5 |
| spce | cpu | spce_skin20_every1 | 4.8982 | 0.0180 | 0.37% | 5 |

The measured LAMMPS loop excludes process startup; `process_wall_seconds` is retained in JSONL separately.

## Workloads and experimental scope

- Workload A: 128,000-atom homogeneous Lennard-Jones melt, 100 steps per CPU segment and 500 steps per GPU segment.
- Workload B: 36,000-atom SPC/E water, bonded interactions plus PPPM long-range electrostatics, 100 steps per segment.
- CPU: independent `build_rl_char/`, MPI 4, one OpenMP thread/rank.
- GPU: `build_kokkos_cuda/`, one RTX 3090, `gpu/aware off`.
- CPU states: idle, pinned CPU contention, and memory-bandwidth contention. GPU states: idle and same-GPU contention.
- Co-runners: four CPU workers pinned to logical CPUs 0--3; a 512 MiB memcpy stressor pinned to CPUs 8--15; and a CUDA kernel on GPU 0.
- Actions were restricted to safe `skin/every` combinations with `check yes`, `delay 0`, and all measured runs had zero dangerous builds.

## State × action performance matrix
### lj / cpu
| State | Action | Mean loop (s) | Std (s) | CV | Throughput | Rank | n |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu_contention | `lj_skin03_every1` | 2.4643 | 0.0052 | 0.21% | 40.58 step/s | 1 | 3 |
| cpu_contention | `lj_skin06_every5` | 2.4703 | 0.0039 | 0.16% | 40.48 step/s | 2 | 3 |
| cpu_contention | `lj_skin10_every20` | 2.5435 | 0.0105 | 0.41% | 39.32 step/s | 3 | 3 |
| idle | `lj_skin03_every1` | 1.7585 | 0.0108 | 0.61% | 56.87 step/s | 1 | 3 |
| idle | `lj_skin06_every5` | 1.7980 | 0.0035 | 0.20% | 55.62 step/s | 2 | 3 |
| idle | `lj_skin10_every20` | 1.8218 | 0.0050 | 0.27% | 54.89 step/s | 3 | 3 |
| memory_contention | `lj_skin03_every1` | 1.8545 | 0.0095 | 0.51% | 53.92 step/s | 1 | 3 |
| memory_contention | `lj_skin06_every5` | 1.8895 | 0.0037 | 0.20% | 52.92 step/s | 2 | 3 |
| memory_contention | `lj_skin10_every20` | 1.9068 | 0.0011 | 0.06% | 52.45 step/s | 3 | 3 |

### lj / gpu
| State | Action | Mean loop (s) | Std (s) | CV | Throughput | Rank | n |
|---|---|---:|---:|---:|---:|---:|---:|
| gpu_contention | `lj_skin10_every20` | 1.9801 | 0.0026 | 0.13% | 252.51 step/s | 1 | 3 |
| gpu_contention | `lj_skin06_every5` | 2.6489 | 0.0008 | 0.03% | 188.76 step/s | 2 | 3 |
| gpu_contention | `lj_skin03_every1` | 4.1360 | 0.0002 | 0.01% | 120.89 step/s | 3 | 3 |
| idle | `lj_skin06_every5` | 0.8472 | 0.0004 | 0.04% | 590.16 step/s | 1 | 3 |
| idle | `lj_skin03_every1` | 0.9045 | 0.0071 | 0.79% | 552.82 step/s | 2 | 3 |
| idle | `lj_skin10_every20` | 0.9518 | 0.0019 | 0.20% | 525.32 step/s | 3 | 3 |

### spce / cpu
| State | Action | Mean loop (s) | Std (s) | CV | Throughput | Rank | n |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu_contention | `spce_skin20_every1` | 7.6011 | 0.0092 | 0.12% | 13.16 step/s | 1 | 3 |
| cpu_contention | `spce_skin30_every5` | 7.6724 | 0.0401 | 0.52% | 13.03 step/s | 2 | 3 |
| cpu_contention | `spce_skin40_every10` | 7.7221 | 0.0101 | 0.13% | 12.95 step/s | 3 | 3 |
| idle | `spce_skin30_every5` | 4.9065 | 0.0284 | 0.58% | 20.38 step/s | 1 | 3 |
| idle | `spce_skin20_every1` | 4.9093 | 0.0252 | 0.51% | 20.37 step/s | 2 | 3 |
| idle | `spce_skin40_every10` | 4.9817 | 0.0143 | 0.29% | 20.07 step/s | 3 | 3 |
| memory_contention | `spce_skin20_every1` | 5.0732 | 0.0079 | 0.16% | 19.71 step/s | 1 | 3 |
| memory_contention | `spce_skin30_every5` | 5.0871 | 0.0023 | 0.05% | 19.66 step/s | 2 | 3 |
| memory_contention | `spce_skin40_every10` | 5.1487 | 0.0171 | 0.33% | 19.42 step/s | 3 | 3 |


## Static controllability

The action response is larger than the idle-run noise for Workload A on both CPU and GPU. For Workload B, the nominal winner changes in some states, but the gaps do not exceed the conservative uncertainty test; this is not yet evidence for state-conditioned CPU action selection.

## Action interaction, relative gaps, and fixed-policy regret
### lj / cpu
| State | Best action | Best mean (s) | Runner-up mean (s) | Gap | Gap exceeds conservative CI |
|---|---|---:|---:|---:|---|
| cpu_contention | `lj_skin03_every1` | 2.4643 | 2.4703 | 0.24% | no |
| idle | `lj_skin03_every1` | 1.7585 | 1.7980 | 2.25% | yes |
| memory_contention | `lj_skin03_every1` | 1.8545 | 1.8895 | 1.88% | yes |

| Fixed-policy statistic | Value |
|---|---:|
| Best global fixed action | `lj_skin03_every1` (2.0258 s mean across states) |
| State-dependent oracle mean | 2.0258 s |
| Global fixed-policy penalty | 0.00% |
| Best-action identity changes | no (lj_skin03_every1, lj_skin03_every1, lj_skin03_every1) |

| State | Action | Regret relative to state oracle |
|---|---|---:|
| cpu_contention | `lj_skin03_every1` | 0.00% |
| cpu_contention | `lj_skin06_every5` | 0.24% |
| cpu_contention | `lj_skin10_every20` | 3.21% |
| idle | `lj_skin03_every1` | 0.00% |
| idle | `lj_skin06_every5` | 2.25% |
| idle | `lj_skin10_every20` | 3.60% |
| memory_contention | `lj_skin03_every1` | 0.00% |
| memory_contention | `lj_skin06_every5` | 1.88% |
| memory_contention | `lj_skin10_every20` | 2.82% |

### lj / gpu
| State | Best action | Best mean (s) | Runner-up mean (s) | Gap | Gap exceeds conservative CI |
|---|---|---:|---:|---:|---|
| gpu_contention | `lj_skin10_every20` | 1.9801 | 2.6489 | 33.78% | yes |
| idle | `lj_skin06_every5` | 0.8472 | 0.9045 | 6.76% | yes |

| Fixed-policy statistic | Value |
|---|---:|
| Best global fixed action | `lj_skin10_every20` (1.4660 s mean across states) |
| State-dependent oracle mean | 1.4137 s |
| Global fixed-policy penalty | 3.70% |
| Best-action identity changes | yes (lj_skin10_every20, lj_skin06_every5) |

| State | Action | Regret relative to state oracle |
|---|---|---:|
| gpu_contention | `lj_skin03_every1` | 108.87% |
| gpu_contention | `lj_skin06_every5` | 33.78% |
| gpu_contention | `lj_skin10_every20` | 0.00% |
| idle | `lj_skin03_every1` | 6.76% |
| idle | `lj_skin06_every5` | 0.00% |
| idle | `lj_skin10_every20` | 12.34% |

### spce / cpu
| State | Best action | Best mean (s) | Runner-up mean (s) | Gap | Gap exceeds conservative CI |
|---|---|---:|---:|---:|---|
| cpu_contention | `spce_skin20_every1` | 7.6011 | 7.6724 | 0.94% | no |
| idle | `spce_skin30_every5` | 4.9065 | 4.9093 | 0.06% | no |
| memory_contention | `spce_skin20_every1` | 5.0732 | 5.0871 | 0.27% | no |

| Fixed-policy statistic | Value |
|---|---:|
| Best global fixed action | `spce_skin20_every1` (5.8612 s mean across states) |
| State-dependent oracle mean | 5.8603 s |
| Global fixed-policy penalty | 0.02% |
| Best-action identity changes | yes (spce_skin20_every1, spce_skin30_every5, spce_skin20_every1) |

| State | Action | Regret relative to state oracle |
|---|---|---:|
| cpu_contention | `spce_skin20_every1` | 0.00% |
| cpu_contention | `spce_skin30_every5` | 0.94% |
| cpu_contention | `spce_skin40_every10` | 1.59% |
| idle | `spce_skin20_every1` | 0.06% |
| idle | `spce_skin30_every5` | 0.00% |
| idle | `spce_skin40_every10` | 1.53% |
| memory_contention | `spce_skin20_every1` | 0.00% |
| memory_contention | `spce_skin30_every5` | 0.27% |
| memory_contention | `spce_skin40_every10` | 1.49% |


## Observable state signals

The table uses the first action in each workload/backend to avoid selecting telemetry after seeing the winner.

| Workload | Backend | State | Mean loop (s) | Pair (s) | Neigh (s) | Kspace (s) | Comm (s) | Mean loadavg before | GPU snapshot before |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| lj | cpu | cpu_contention | 2.4643 | 1.7157 | 0.6365 | - | 0.0522 | 2.53 | `0, 0, 0, 1, 24576; 1, 0, 0, 1, 24576; 2, 0, 0, 1, 24576; 3, 0, 0, 1, 24576` |
| lj | cpu | idle | 1.7585 | 1.1950 | 0.4663 | - | 0.0471 | 2.04 | `0, 0, 0, 1, 24576; 1, 0, 0, 1, 24576; 2, 0, 0, 1, 24576; 3, 0, 0, 1, 24576` |
| lj | cpu | memory_contention | 1.8545 | 1.2577 | 0.4806 | - | 0.0556 | 4.43 | `0, 0, 0, 1, 24576; 1, 0, 0, 1, 24576; 2, 0, 0, 1, 24576; 3, 0, 0, 1, 24576` |
| lj | gpu | gpu_contention | 4.1360 | 0.0196 | 0.8797 | - | 2.2411 | 3.49 | `0, 0, 0, 1, 24576; 1, 0, 0, 1, 24576; 2, 0, 0, 1, 24576; 3, 0, 0, 1, 24576` |
| lj | gpu | idle | 0.9045 | 0.0143 | 0.3556 | - | 0.0768 | 4.82 | `0, 0, 0, 1, 24576; 1, 0, 0, 1, 24576; 2, 0, 0, 1, 24576; 3, 0, 0, 1, 24576` |
| spce | cpu | cpu_contention | 7.6011 | 5.5155 | 0.9306 | 0.9881 | 0.0376 | 4.71 | `0, 0, 0, 1, 24576; 1, 0, 0, 1, 24576; 2, 0, 0, 1, 24576; 3, 0, 0, 1, 24576` |
| spce | cpu | idle | 4.9093 | 3.5971 | 0.5893 | 0.5914 | 0.0323 | 3.91 | `0, 0, 0, 1, 24576; 1, 0, 0, 1, 24576; 2, 0, 0, 1, 24576; 3, 0, 0, 1, 24576` |
| spce | cpu | memory_contention | 5.0732 | 3.7025 | 0.6046 | 0.6148 | 0.0389 | 6.60 | `0, 0, 0, 1, 24576; 1, 0, 0, 1, 24576; 2, 0, 0, 1, 24576; 3, 0, 0, 1, 24576` |

## External nonstationarity

CPU contention increased the LJ CPU loop time from roughly 1.76--1.82 s to 2.46--2.54 s and the SPC/E CPU loop time from roughly 4.90--4.98 s to 7.60--7.72 s. Memory contention was weaker but measurable (about 5% for LJ and 3--4% for SPC/E). Same-GPU contention increased the LJ GPU loop time by approximately 2.1--4.6× depending on the action.

## Time-varying episodes

The segmented probe keeps one LAMMPS state alive and changes only the external co-runner between 20-step segments. LAMMPS loop time excludes Python/library and co-runner startup; those process-level times remain in the JSONL.

| Workload | State | n | Mean loop (s) | Std (s) | CV | Pair (s) | Neigh (s) | Kspace (s) | Comm (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lj | cpu_contention | 2 | 2.5954 | 0.0748 | 2.88% | 1.8987 | 0.6072 | - | 0.0183 |
| lj | idle | 4 | 1.1940 | 0.0632 | 5.29% | 0.8702 | 0.2803 | - | 0.0103 |
| lj | memory_contention | 2 | 1.2574 | 0.1084 | 8.62% | 0.8987 | 0.2922 | - | 0.0148 |
| spce | cpu_contention | 2 | 5.7848 | 2.7683 | 47.86% | 4.6788 | 0.3801 | 0.6237 | 0.0114 |
| spce | idle | 4 | 3.7700 | 0.1833 | 4.86% | 2.9519 | 0.3580 | 0.3933 | 0.0091 |
| spce | memory_contention | 2 | 3.9032 | 0.0398 | 1.02% | 3.1371 | 0.2503 | 0.4264 | 0.0136 |

The two orders make contention changes observable in the workload-level timing, but the second SPC/E CPU-contention segment did not slow down as much as the first. This indicates that the simple synthetic co-runner is useful but not perfectly repeatable at this short segment length; static matrix conclusions therefore carry more weight than this probe alone.

### Idle-only internal-phase probe

The following is four consecutive idle segments in one LAMMPS process, using the baseline action. Segment 0 is retained to expose warm-up; the post-warm-up CV is calculated over segments 1--3.

| Workload | n | Segment 0 loop (s) | Segments 1--3 mean (s) | Segments 1--3 CV | Segment 3 loop (s) | Temp 0 → 3 |
|---|---:|---:|---:|---:|---:|---:|
| lj | 1 | 1.0955 | 1.2296 | 0.38% | 1.2246 | 0.63 → 0.76 |
| spce | 1 | 3.5818 | 3.7205 | 3.24% | 3.8581 | 248.36 → 270.83 |

The idle-only probe shows natural temperature evolution, but no sustained monotonic Pair/Kspace change that would by itself establish a new performance regime. The observed LJ first-segment warm-up and the SPC/E final neighbor rebuild are measurement-phase effects, not evidence that the optimal action changes. A longer naturally evolving or spatially imbalanced workload is still needed before treating internal phase as a control state.

## Recommended future environment

Retain application timing (`Pair`, `Neigh`, `Kspace`, `Comm`, `Modify`, loop time), neighbor-build counts, backend/resource configuration, and low-overhead load/GPU telemetry. On CPU, discard `skin/every` as a first RL action unless a larger workload or a wider valid range reveals a larger interaction; the tested CPU optimum was effectively constant. On GPU, retain the three neighbor actions and the contention/resource telemetry. Keep MPI rank count, OpenMP count, backend, and GPU mapping as episode-level configuration, not online actions.

For an initial fixed-duration environment, use 100 LAMMPS steps per CPU intervention and 500 steps per GPU intervention: these are the stable matrix settings. The 20-step library probe is useful for observing transitions but is too short/noisy for the SPC/E control interval.

For the current evidence, online action adaptation is supported for the GPU under external resource contention, while CPU neighbor-action adaptation is not yet justified. The changing episode shows multi-second state changes, so SMDP duration is plausible; however, the short synthetic episode does not yet establish the best variable intervention policy or interval.

## Limitations

The CPU co-runners are synthetic local stressors, the memory stressor is a simple memcpy loop, and the GPU contender shares GPU 0 only. Measurements were intentionally small and not a production scaling study. The current Open MPI is not CUDA-aware, so `gpu/aware on` was excluded.
