# Candidate runtime control parameters

This is a design inventory, not an automatic tuner.  The first environment
should use a small, validated action space and should reject values outside the
workload-specific safe range.  Paths below are relative to the LAMMPS source
tree at this commit.

## A. Safer performance candidates

These controls primarily change scheduling, decomposition, or bookkeeping.  A
candidate is only considered safe when the required neighbor/communication
coverage is preserved and scientific output requirements are unchanged.

| Mechanism | Expected effect and initial range | Change timing | Semantics | Documentation / implementation |
|---|---|---|---|---|
| `neigh_modify every N delay D check yes` | Neighbor-list rebuild frequency; for this LJ workload start with `N=1..40`, `D=0..10`, and `check yes`. Larger values reduce rebuild work but can increase risk of stale lists if used carelessly. | Between segmented `run` commands; changing `every/delay/check` during a run should be treated as a future `run every` experiment, not assumed. | Pair lists still use the configured cutoff and skin; `check yes` is the conservative default. | `doc/src/neigh_modify.rst`; `src/neighbor.cpp` |
| `neighbor skin bin` | Skin controls extra neighbor-list distance; for this LJ input a bounded trial range is `0.2..0.8` in LJ units. Larger skins rebuild less often but inspect more candidates. | Set before a run segment; it may force list reinitialization. | No force cutoff change, but too-small skins can produce dangerous builds and should be rejected. | `doc/src/neighbor.rst`; `src/neighbor.cpp` |
| `comm_modify` | Ghost-atom communication mode/cutoff and velocity exchange; useful for multi-style or communication-heavy workloads. Start with the documented default and only non-underestimating cutoffs. | Usually between segments; some options are setup-sensitive. | Safe only when the ghost region remains large enough for every active interaction. | `doc/src/comm_modify.rst`; `src/comm.cpp` |
| `balance` / `fix balance` | Redistributes subdomains when atom, neighbor, or timing load is imbalanced. Start with conservative thresholds such as `1.05..1.30`. | `balance` between segments; `fix balance` can intervene during a run but changes domain decomposition at its chosen frequency. | Intended as a performance/load-balance operation; ordering and floating-point reduction order can still change. | `doc/src/balance.rst`; `src/balance.cpp`; `src/fix_balance.cpp` |
| `atom_modify sort N binsize` | Local atom reorder frequency and bin size can improve cache locality and pair traversal. | Prefer between segments or at initialization. | Performance/bookkeeping only, but floating-point order can differ. | `doc/src/atom_modify.rst`; `src/atom.cpp` |
| `thermo N`, `thermo_style`, `thermo_modify` | Thermodynamic output frequency and formatting; reduces I/O/formatting overhead when output is not part of the observation. | Between segments. | No physical change; reducing observations changes what the controller can see. | `doc/src/thermo.rst`; `src/thermo.cpp` |
| `dump` / `restart` frequency | Output and checkpoint overhead, especially relevant because `/work` is HDD-backed. | Between segments, or with a fixed schedule in the input. | No physical change, but less output can remove required scientific records. | `doc/src/dump.rst`; `doc/src/restart.rst`; `src/output.cpp` |
| `run N` segment length | SMDP duration/action-hold time: controls when the controller observes and can intervene again. Initial range `100..5000` for this workload. | This is the intervention boundary itself. | Does not change the equations if commands remain otherwise identical. | `doc/src/run.rst`; `src/run.cpp` |
| `OMP_NUM_THREADS`, `package omp N`, `-sf omp` | Thread/rank placement and use of OpenMP accelerated styles. On this host start with one thread per rank, then test bounded rank/thread products without oversubscription. | Process environment before launch; package/suffix settings normally require initialization or a new segment/setup. | OpenMP styles may alter reduction order slightly; compare validation observables. | `doc/src/package.rst`; `doc/src/Speed_omp.rst`; `src/OPENMP/fix_omp.cpp`; `cmake/Modules/Packages/OPENMP.cmake` |
| `-sf kk -k on g Ng` with a fixed rank/device map | Selects the KOKKOS CUDA implementation and the number of GPUs per MPI rank. Start with one rank per RTX 3090 and `Ng=1`; use only prevalidated mappings. | Launch/setup time; changing this requires a new executable invocation or a deliberately designed restart boundary. | Implementation choice; compare thermo/energy observables because reduction order and available styles can differ. | `doc/src/Speed_kokkos.rst`; `doc/src/package.rst`; `src/KOKKOS/kokkos.cpp` |
| `-pk kokkos gpu/aware off` | Uses host-staged MPI communication rather than passing device buffers to MPI. It is the safe current setting on this host and avoids the unsupported CUDA-aware MPI path. | Launch/setup time; set before the run or segment sequence. | Communication implementation only, subject to validation; it does not alter the physical input. | `doc/src/Speed_kokkos.rst`; `doc/src/package.rst`; `src/KOKKOS/comm_kokkos.cpp` |

`processors Px Py Pz` is also a high-value decomposition candidate for larger
systems, but it is normally an initialization-time choice.  On this host it
should be evaluated together with NUMA affinity because GPUs 0--2 are on node 0
and GPU 3 is on node 1.  See `doc/src/processors.rst`, `src/comm.cpp`, and
`src/procmap.cpp`.

## B. Conditional candidates

These can improve runtime but require explicit scientific validation or a
reinitialization.

| Mechanism | Why conditional | Change timing / range | References |
|---|---|---|---|
| `newton on/off` | Changes force/communication ownership and can change floating-point reduction order; some styles impose restrictions. | Initialization or carefully separated segments; only values accepted by the active styles. | `doc/src/newton.rst`; `src/force.cpp`; `src/comm.cpp` |
| `suffix omp`, `suffix gpu`, `suffix kk`, `-sf` | Selects alternate implementations. Results should be checked against the unsuffixed style; the CUDA KOKKOS executable is separate from the CPU baseline. | Executable/build and initialization choice; changing the suffix requires reinitialization or a new run. | `doc/src/suffix.rst`; `doc/src/Speed_kokkos.rst`; `cmake/Modules/Packages/KOKKOS.cmake` |
| `package gpu` / GPU device selection | Can offload pair/neighbor work. This host has 4 RTX 3090 cards but no NVLink and reports no P2P-read support between pairs, so device/rank mapping matters. | Launch/setup time; keep device count and mapping explicit. | `doc/src/Speed_gpu.rst`; `doc/src/package.rst`; `GPU/fix_gpu.cpp` |
| `-pk kokkos gpu/aware on` | Requires MPI to support CUDA device buffers. The current Open MPI was not built with CUDA support; the four-GPU probe segfaulted, so this is not an allowed action until MPI is replaced/rebuilt and revalidated. | Launch/setup time; do not toggle blindly during a run. | `doc/src/Speed_kokkos.rst`; `doc/src/package.rst`; Open MPI `ompi_info --all` |
| `comm_modify cutoff` below the required ghost extent | Reducing communication can look faster but drops required interactions and changes the simulation. | Never choose automatically without deriving a lower bound from active styles. | `doc/src/comm_modify.rst`; `src/comm.cpp` |
| `timestep` | Directly changes the integration problem and trajectory/accuracy. | Scientific parameter, not a performance action in the first environment. | `doc/src/timestep.rst`; `src/update.cpp` |
| `fix`/thermostat/barostat frequency, `kspace` accuracy | Can alter trajectory, ensemble, or numerical accuracy even if wall time improves. | Conditional, with domain-specific validation. | Relevant `doc/src/fix_*.rst`, `doc/src/kspace_style.rst` |
| Pair style, cutoff, force-field, atom count, and `velocity` seed | Changes the scientific workload or its initial condition. | Keep fixed for the baseline and first controller. | Relevant pair-style, `region`, `create_atoms`, and `velocity` documentation |

## Practical action-space recommendation

Start with three discrete/categorical dimensions and a duration action:

1. `neigh_modify every` in `{1, 5, 10, 20}` with `delay=0`, `check=yes`.
2. MPI process count in a prevalidated set such as `{1, 2, 4, 8}`; treat this
   as a launch-level configuration, not a mid-run mutation.
3. OpenMP threads per rank in a prevalidated set, with the product bounded by
   available cores and explicit affinity.
4. Segment duration `Δt` in a small set such as `{100, 500, 1000}` steps.

For the current input, the bundled benchmark uses `check no` and `every 20`.
That setting is retained in `bench_lj.in` for comparability, but an automated
controller should first validate `check yes` candidates and reject any run
with dangerous neighbor builds or unacceptable observable drift.
