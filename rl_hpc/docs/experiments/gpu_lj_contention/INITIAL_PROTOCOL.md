# GPU and network contention characterization

Status: instrumentation and independent builds prepared on 2026-09-22, but
the new runtime matrices were not fabricated because this shell currently has
no usable NVIDIA device nodes and no multi-node allocation.

## Executive conclusion

The earlier three-action GPU result remains useful as a hypothesis: same-GPU
contention changed the nominal LJ winner from `skin=0.6,every=5` to
`skin=1.0,every=20`, with a 3.70% best-fixed versus state-oracle gap. It does
not identify whether `skin`, `every`, or their interaction caused that result.
The new 3x3 factorial is therefore required before making a mechanism claim.

The current machine cannot run that factorial. `nvidia-smi` reports that it
cannot communicate with the NVIDIA driver, `/dev/nvidia*` is absent, and a
direct CUDA contender probe fails at `cudaMalloc`. PCI inspection still sees
four RTX 3090 controllers with the NVIDIA kernel module listed, so this is a
runtime/device exposure failure rather than evidence about LAMMPS performance.

No genuine network matrix was run. The host has a reachable 10 Gb/s Intel X550
Ethernet interface and can ping other site hosts, but there is no scheduler
allocation, SSH authentication for remote MPI daemons, or valid multi-node
launch. `mpirun --host amp1,amp4 -np 2 hostname` fails before starting a
remote daemon. Localhost traffic was intentionally not used as an HPC network
proxy.

## GPU experiment prepared

`characterization/gpu_factorial.py` measures the complete action grid:

| Parameter | Values |
|---|---|
| `skin` | 0.3, 0.6, 1.0 |
| `neigh_modify every` | 1, 5, 20 |
| `delay` | 0 |
| `check` | yes |
| target GPU | physical GPU 0 |
| CPU affinity | logical CPUs 0--3, consistently applied to target |
| stable LJ segment | 500 steps |
| default repetitions | 5, randomized order per state/repetition |

The prepared states are:

- `idle`
- `same_gpu_light`, `same_gpu_medium`, `same_gpu_heavy`
- `different_gpu_medium` with the contender on physical GPU 1
- `cpu_only` with four CPU workers pinned to CPUs 0--3

The GPU contender intensity is explicit and reproducible:

| Intensity | Array size per buffer | Kernel repeats | Sleep |
|---|---:|---:|---:|
| light | 64 MiB | 8 | 1000 microseconds |
| medium | 128 MiB | 32 | 0 |
| heavy | 256 MiB | 64 | 0 |

Each record preserves action, mapping, contender configuration, LAMMPS
timings, neighbor builds, raw `nvidia-smi` samples, clocks, power,
temperature, utilization, memory usage, exit status, and exact commands.
`analyze_gpu_factorial.py` produces the full matrix, regret, fixed-policy
oracle comparison, and simple skin/every marginal and interaction summaries
once JSONL data are available.

The direct command for the full LJ matrix is:

```bash
python3 rl_hpc/characterization/gpu_factorial.py \
  --output "rl_hpc/characterization/data/gpu-factorial-$(date +%Y%m%d-%H%M%S)" \
  --workload lj --gpu-build build_kokkos_cuda --steps 500 --reps 5
```

The current runtime blocker must be resolved before this command can produce
scientific results.

## SPC/E GPU follow-up prepared

The existing `build_kokkos_cuda/` contains KOKKOS only. A separate
`build_kokkos_cuda_char/` was configured and compiled successfully with:

- KOKKOS CUDA/Serial
- KSPACE
- MOLECULE
- EXTRA-MOLECULE
- RIGID
- CUDA 13.4.59, C++20, `AMPERE86`, MPI, `gpu/aware off`

This preserves the original CUDA build. After GPU recovery, the intended
compact SPC/E follow-up is:

```bash
python3 rl_hpc/characterization/gpu_factorial.py \
  --output "rl_hpc/characterization/data/gpu-spce-subset-$(date +%Y%m%d-%H%M%S)" \
  --workload spce --gpu-build build_kokkos_cuda_char --steps 100 \
  --reps 5 \
  --actions spce_skin20_every1 spce_skin30_every5 spce_skin40_every10 \
  --states idle same_gpu_medium same_gpu_heavy different_gpu_medium
```

This is deliberately a subset rather than a second 9-action sweep.

## Network environment and experiment path

Details are in [the network environment record](../../environment/NETWORK.md). The
observed facts are:

- active interface: `enp1s0f1`, Intel X550, 10 Gb/s, `10.132.10.16/16`;
- no `srun`, `sbatch`, or `qsub` in the current shell;
- no local InfiniBand verbs device detected;
- `iperf3` and Open MPI are installed;
- `amp1`/`amp4` respond to ICMP, but SSH authentication and remote MPI daemon
  startup fail;
- no traffic was sent to unallocated nodes.

`network_corunner/mpi_stream.c` is a bounded MPI ring traffic generator with
message size, stream count, duration, and duty-cycle controls. It compiled
successfully with `mpicc -O3 -std=c11`. `network_runner.py` requires an
explicit host list and supports N0--N3 traffic levels, but was only dry-run
validated because no allocated host list is available. A two-rank localhost
smoke test verified clean MPI operation and termination; its throughput was
not used as a network-contestation measurement.

The intended network target is SPC/E + PPPM, retaining `Comm` and `Kspace`
timings. The N0--N3 action matrix, fixed/oracle comparison, and mechanism
analysis are pending a genuine multi-node allocation.

## Answers to the research questions

### GPU

1. The prior GPU result is reproducible only at the already-measured
   three-action level; the requested separated 3x3 replication was blocked by
   the unavailable driver.
2. Same-GPU causality is not yet proven. Different-GPU and CPU-only controls
   are instrumented but unmeasured in this run.
3. The intensity response is not yet measured; raw telemetry support is ready.
4. The skin/every mechanism is not identifiable from the coupled three-action
   experiment. The factorial analyzer is ready to separate marginal and
   interaction effects.
5. SPC/E GPU behavior is not yet measured. The required independent build is
   ready.
6. The previously measured fixed-policy gap remains approximately 3.70%, but
   it is not an answer to the separated-factor question.

### Network

7. Genuine multi-node execution was not available from the current shell.
8. Network impact on LAMMPS is therefore unmeasured.
9. `Comm`/`Kspace` mechanism changes are unmeasured.
10. Network-conditioned optimal action and fixed-policy regret are unmeasured.
11. SPC/E + PPPM is the prepared target; LJ is not required for the first
    network experiment.

### Research conclusion

The strongest currently supported candidate remains same-GPU contention for
GPU LJ, because it is the only source with an observed non-negligible
fixed-policy regret. The state vector should retain application timing,
neighbor-build counts, GPU utilization/memory/clocks/power/temperature, and
explicit resource mapping. The first action candidate remains the neighbor
grid, but `skin` versus `every` must not be collapsed until the factorial is
run. MPI rank count, GPU mapping, and backend remain episode-level settings.

For the first fixed-duration baseline, retain the previously stable 500 GPU
steps (and 100 CPU steps). No claim about SMDP duration is made here.

## Files added for this task

- `rl_hpc/CONTENTION_CHARACTERIZATION.md`
- `rl_hpc/docs/environment/NETWORK.md`
- `rl_hpc/build_kokkos_cuda_characterization.sh`
- `rl_hpc/characterization/gpu_contender.cu` (intensity controls)
- `rl_hpc/characterization/gpu_factorial.py`
- `rl_hpc/characterization/analyze_gpu_factorial.py`
- `rl_hpc/characterization/network_runner.py`
- `rl_hpc/characterization/network_corunner/mpi_stream.c`

No RL, SMDP, adaptive policy, or scientific-parameter tuning was added.
