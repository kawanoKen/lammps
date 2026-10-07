# Network environment for contention characterization

Inspected on 2026-09-22 from `amp2.g2.gsic.titech.ac.jp`.

## What is available

- No Slurm/PBS launcher was present: `srun`, `sbatch`, and `qsub` were not found.
- The active interface is `enp1s0f1`, an Intel X550 10 Gb/s Ethernet interface,
  with address `10.132.10.16/16` and a default route through `10.132.0.1`.
- `enp1s0f0` is down. No InfiniBand verbs device was reported by the local
  probe, and `lspci` shows the X550 Ethernet controllers rather than an
  InfiniBand/HDR adapter.
- Open MPI 5.0.10 provides `pml/ob1`, `pml/ucx`, and TCP/OFI/shared-memory
  components. The active inter-node transport cannot be identified without a
  successful multi-node MPI launch. CUDA-aware MPI remains disabled.
- `iperf3` and `mpirun` are installed locally.
- Other site hosts, including `amp1` and `amp4`, respond to ICMP from this
  node. This establishes IP reachability, not permission to use them for a
  workload.

## Why a genuine network experiment could not run

The current shell has no scheduler allocation containing multiple nodes. SSH
to `amp1`, `amp4`, and `amp5` with the current user failed with public-key or
password authentication errors. A read-only probe using
`mpirun --host amp1,amp4 -np 2 hostname` consequently failed while spawning
the remote MPI daemon (status 255). No network traffic generator was started
against another host.

Running `iperf3` against localhost, or using two local MPI ranks, would not be
an HPC inter-node contention result and was intentionally not used as a proxy.
The network part therefore requires a real multi-node allocation or valid
remote launch credentials.

## Prepared experiment path

`rl_hpc/characterization/network_corunner/mpi_stream.c` is a bounded MPI
ring-stream generator. It supports message size, stream count, duration, and
inter-iteration sleep controls, reports aggregate throughput, and exits cleanly
on SIGTERM. `network_runner.py` is prepared to launch the traffic generator and
an MPI LAMMPS target within an explicitly supplied host list. Both scripts
require the caller to provide allocated hosts; neither discovers or uses
unallocated site nodes.

When a real allocation is available, first record the launcher and host list,
then validate N0 with traffic disabled and N1--N3 with measured co-runner
throughput. The target should use SPC/E + PPPM and retain its `Comm` and
`Kspace` timing breakdown.
