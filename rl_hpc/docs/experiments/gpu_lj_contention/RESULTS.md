# Single-node non-stationarity characterization

Captured on 2026-09-22. No RL or SMDP implementation was added.

## GPU mechanism

### Clean independent-process recapture

`characterization/gpu_clean_protocol.py` was added for this recapture. It
runs each LAMMPS phase in a separate process, waits for GPU 0 to return below
45 C, 45 W, and 2% utilisation before the next phase, and records telemetry
before, during, and after every phase. It flags samples reaching 345 W as
power-capped and reaps the CUDA co-runner process group on exit.

Two safe actions were measured in randomized-independent A→X→A brackets at
1,000 steps: `skin=0.6,every=5` and `skin=1.0,every=20`. A is idle, B is a
co-runner on GPU 0, C on GPU 1 (`NODE`, same NUMA node), and D on GPU 3
(`SYS`, cross socket).

| Action | A baseline range (s) | B / GPU 0 (s) | C / GPU 1 (s) | D / GPU 3 (s) |
|---|---:|---:|---:|---:|
| skin=0.6, every=5 | 1.737--1.756 | 3.816, 3.815, 3.813 | 1.757 | 1.743 |
| skin=1.0, every=20 | 1.919--1.937 | 3.144, 3.145, 3.133 | 1.924 | 1.927 |

The second B repetition for each action reached the 345 W power-cap flag,
whereas every A/C/D phase and the reverse-order third B repetition were below
that threshold. Despite this flag, the three B repetitions agree to within
0.4% per action and the A brackets return
to baseline. The clean data therefore reproduce the interaction: idle favors
`skin=0.6,every=5` (about 10% faster), but direct GPU-0 contention favors
`skin=1.0,every=20` (about 18% faster). GPU 1 and GPU 3 co-runners have
negligible effect, so the mechanism is direct GPU execution contention rather
than same-NUMA or cross-socket shared-host effects.

The current clean two-state, two-action fixed-policy/oracle gap is about
3.8%: choosing the idle action under B loses roughly 21%, while choosing the
contention action under idle loses roughly 11%. This is a small recapture,
not a replacement for a 7--10 repetition factorial. The protocol records a
UTC timestamp so the same actions can be rerun on later days; multi-day data
are not yet available.

`nvidia-smi topo -m` shows GPUs 0--2 connected by `NODE` paths in NUMA node 0; GPU 3 is `SYS` from GPU 0 and crosses the socket interconnect. There is no NVLink. The target is GPU 0 on CPUs 0--3; GPU co-runners use CPUs 8--11 to avoid direct CPU-core contention.

The 1,000-step GPU-LJ idle calibration (`skin=0.6,every=5`) completed five times: 1.74555, 1.75390, 1.72298, 1.74782, and 1.72589 s. Mean: 1.73923 s; sample std: 0.01388 s; CV: 0.80%. This is an adequate isolated segment length.

The complete 3x3 grid was started with `delay=0,check=yes`. `skin=0.3,every=20` and sometimes `skin=0.6,every=20` produced dangerous neighbor builds, so those cells are explicitly excluded. The long factorial was not valid: GPU 0 reached the 350 W software power cap and nominally idle timing drifted from about 2 s to over 8 s. A continuation timeout also left a CUDA co-runner orphaned; it was identified and terminated. GPU 0 then fell from 75 C / 350 W / 100% utilisation to 61 C / 116 W / 0%.

The partial factorial data are audit records only: they cannot distinguish action effects from power/thermal history. The prior three-action result remains the only valid interaction evidence: idle best `skin=0.6,every=5`, same-GPU contention best `skin=1.0,every=20`, fixed-policy/oracle gap about 3.7%. The separated-factor reproduction is inconclusive and should be rerun in a batch job with process containment and a cooling/power-state protocol.

## Communication emulation

No multi-node allocation exists. A local test was therefore labeled communication emulation, not HPC-network contention. Open MPI was forced away from shared memory:

```bash
mpirun --mca btl tcp,self --mca btl_tcp_if_include lo -np 4 ...
```

Verbose output confirms `Using interface: lo` and `Using tcp btl for send`. SPC/E+PPPM completed 100 steps in 5.03943 s: Pair 3.8811 s (77.01%), Kspace 0.65314 s (12.96%), Neigh 0.34064 s (6.76%), Comm 0.054325 s (1.08%).

An isolated namespace for `tc netem` was not permitted: `unshare -Urn` failed with `write failed /proc/self/uid_map: Operation not permitted`. No global network change was attempted. Hence there are no controlled degraded-communication states, no communication state-action matrix, and no communication-conditioned regret result. With Comm at only 1.08% in this smoke test, genuine multi-node validation is not yet the highest-priority next experiment.

## NUMA and memory contention

`numactl -H` reports node 0 CPUs 0--23,48--71 (385364 MB) and node 1 CPUs 24--47,72--95 (382904 MB); local/remote distances are 10/21. The earlier CPU matrix already used target ranks on node-0 cores and a 512 MiB memcpy co-runner on node-0 CPUs 8--15. Same-node memory contention raised LJ best-action time 1.7585 to 1.8545 s (5.5%) and SPC/E 4.9093 to 5.0732 s (3.3%), but did not change a statistically useful neighbor action. Remote-node contention was not run after the GPU runner cleanup issue; this is pending, not a negative result.

## Conclusions

- Observable state change: yes for CPU/memory and earlier same-GPU contention.
- Demonstrated state-conditioned action: only prior GPU LJ (3.7% gap); the new factorial does not yet attribute it to `skin`, `every`, or their interaction.
- Retain state: LAMMPS loop and Pair/Neigh/Comm/Kspace/Modify times, neighbor/dangerous builds, GPU utilization/clock/power/temperature/memory, CPU affinity and NUMA mapping.
- Retain action: validated `skin` and `neigh_modify every` only, with `delay=0`, `check=yes`, and dangerous-build rejection.
- Drop for now: CPU neighbor actions, communication controls, and `gpu/aware on`.

## Artifacts

- `characterization/gpu_factorial.py` now resolves a build directory to `lmp` and pins the CUDA launch thread away from target cores.
- `characterization/data/single-node-gpu-calibration-v2/` holds valid calibration JSONL.
- `characterization/data/single-node-comm-tcp-smoke/log.lammps` holds the forced-TCP smoke log.

Partial GPU data remain in ignored `characterization/data/` for auditability and must not be analysed as a valid factorial matrix.
