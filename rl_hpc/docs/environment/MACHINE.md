# Environment record

Captured on 2026-09-22 on `amp2.g2.gsic.titech.ac.jp` from repository
`/work/kawano/lammps`.

## Repository

- Branch: `develop`
- Commit: `c8bd2ae5927ee236a8892dbd51c18a92cc9c33cf`
- LAMMPS version reported by CMake: `2026.9.2.0`
- Working tree was clean before the `rl_hpc/` additions.

## Operating system and hardware

- OS: Ubuntu 26.04.1 LTS (Resolute Raccoon)
- Kernel: Linux 7.0.0-31-generic, x86_64
- Platform: TYAN B7105F48TV4HR-2T-N / S7105AGM2NR-2T
- CPU: 2 x Intel Xeon Gold 6240R, 24 cores/socket, 2 threads/core
- CPU topology: 48 physical cores, 96 logical CPUs, 2 NUMA nodes
- CPU frequency range reported by `lscpu`: 1.0--4.0 GHz; nominal 2.4 GHz
- CPU features relevant to LAMMPS: AVX2, AVX-512, AVX-512 VNNI, FMA
- Memory: 768 GB installed, approximately 750 GiB available to Linux
- Memory modules: 12 x 64 GB Samsung DDR4-2933, single-bit ECC reported
- NUMA memory: approximately 384 GB per node
- Swap: 2 GiB, unused at inspection time
- GPUs: 4 x NVIDIA GeForce RTX 3090, 24 GiB each, compute capability 8.6
- NVIDIA driver: 615.71.09; CUDA runtime/toolkit: 13.4
- GPU topology: GPUs 0--2 are attached to NUMA node 0; GPU 3 to node 1.
  No NVLink is reported. `nvidia-smi topo -p2p r` reported `GNS` for every
  GPU pair on this host.
- Storage: Samsung approximately 960 GB SSD for `/`; two HGST approximately
  12 TB disks for `/home` and `/work`. `/work` had approximately 7.5 TB free.
- Network: Intel X550, active interface at 10,000 Mb/s full duplex.
- CPU governor: `powersave` at inspection time.

## Software toolchain

The following versions were detected before building:

```text
cmake 4.2.3
GNU gcc/g++ 15.2.0
Open MPI 5.0.10 (mpicc wrapper; MPI libraries found by CMake)
Python 3.14.4
CUDA toolkit 13.4.59 (`nvcc`)
NVIDIA driver 615.71.09
```

The system has `/usr/bin/mpirun` and `/usr/bin/mpiexec`; the selected launcher
is Open MPI's `/usr/bin/mpirun.openmpi`.  Ninja was not available as an
executable, so the build uses GNU Make through the `Unix Makefiles` CMake
generator.

## LAMMPS build

The successful build is in the ignored out-of-source directory `build_rl/`:

```text
CMAKE_BUILD_TYPE=Release
CMAKE_C_COMPILER=/usr/bin/gcc
CMAKE_CXX_COMPILER=/usr/bin/g++
CMAKE_CXX_FLAGS=-march=native
BUILD_MPI=ON
BUILD_OMP=ON
PKG_OPENMP=ON
PKG_PYTHON=ON
BUILD_SHARED_LIBS=ON
CMAKE_INSTALL_PREFIX=/work/kawano/lammps/build_rl/install
```

The build generated `build_rl/lmp` and `build_rl/liblammps.so`.  No system-wide
LAMMPS or Python package installation was performed.

The equivalent configure/build commands are:

```bash
cmake --fresh -S cmake -B build_rl -G 'Unix Makefiles' \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ \
  -D CMAKE_CXX_FLAGS=-march=native \
  -D BUILD_MPI=ON -D BUILD_OMP=ON -D PKG_OPENMP=ON \
  -D PKG_PYTHON=ON -D PYTHON_EXECUTABLE=/usr/bin/python3 \
  -D BUILD_SHARED_LIBS=ON \
  -D CMAKE_INSTALL_PREFIX="$PWD/build_rl/install"
cmake --build build_rl --parallel 24
```

`bash rl_hpc/build.sh` is the preferred copy-pastable form; it selects the
same generator on this host and allows `BUILD_DIR`, `BUILD_JOBS`, `CC`, `CXX`,
and `PYTHON_EXECUTABLE` to be overridden.

## Independent CUDA/KOKKOS build

The CPU build above was left intact.  The separately configured and built
directory `build_kokkos_cuda/` uses:

```text
CMAKE_BUILD_TYPE=Release
CMAKE_C_COMPILER=/usr/bin/gcc
CMAKE_CXX_COMPILER=/usr/bin/g++
CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
CMAKE_CXX_STANDARD=20
BUILD_MPI=ON
BUILD_SHARED_LIBS=OFF
PKG_KOKKOS=ON
Kokkos_ENABLE_SERIAL=ON
Kokkos_ENABLE_CUDA=ON
Kokkos_ARCH_SKX=ON
Kokkos_ARCH_AMPERE86=ON
```

The exact reproducible command is in `rl_hpc/build_kokkos_cuda.sh`:

```bash
bash rl_hpc/build_kokkos_cuda.sh
```

The successful configuration used Kokkos 5.2.1, CUDA 13.4.59, C++20, and
Open MPI 5.0.10.  `build_kokkos_cuda/lmp -h` reports `KOKKOS package API:
CUDA Serial`, double precision, and Kokkos library version 5.2.1.  NVIDIA's
CUDA compiler emits several Kokkos legacy-view host/device diagnostic warnings
during compilation, but the build completes and the executable runs.

Open MPI reports:

```text
opal_built_with_cuda_support: false
opal_cuda_support: false
```

Therefore KOKKOS multi-GPU runs must currently use `-pk kokkos gpu/aware off`.
An explicit four-rank `gpu/aware on` probe was run only as a diagnostic and
segfaulted in MPI device-buffer communication (exit status 139), as expected
for this non-CUDA-aware MPI build.

During the later contention characterization session, the PCI devices and
NVIDIA kernel modules were visible but `/dev/nvidia*` was absent. Both
`nvidia-smi` and a direct CUDA contender reported that the driver could not be
contacted / `cudaMalloc` failed. No driver reload or system-level repair was
attempted; this blocked new GPU runtime measurements without changing the
machine configuration.

## Independent characterization CPU build

The original `build_rl/` directory was not changed. The workload/state/action
characterization uses a separate `build_rl_char/` configured by
`rl_hpc/build_characterization_cpu.sh` with the same GCC/MPI/OpenMP/Python
baseline plus `KSPACE`, `MOLECULE`, `EXTRA-MOLECULE`, and `RIGID` for the bundled
SPC/E/PPPM input. It is a Release shared build with `-march=native` and GNU
Make. `build_rl_char/lmp -h` reports the expected packages, and all 82 static
matrix/stability trials completed with zero dangerous neighbor builds.

For the GPU SPC/E follow-up, `rl_hpc/build_kokkos_cuda_characterization.sh`
created a separate `build_kokkos_cuda_char/` with KOKKOS CUDA/Serial plus
KSPACE, MOLECULE, EXTRA-MOLECULE, and RIGID. Its help output reports all five
packages. Runtime validation is pending because the current shell has no
usable `/dev/nvidia*` device nodes.

## Validation snapshot

- `build_rl/lmp -h`: successful; reports Git info for `develop / c8bd2ae592`.
- Bundled `bench/in.lj`: successful with 1 MPI process and 4 MPI processes.
- The 32,000-atom, 100-step bundled input reported approximately 1.34 s and
  0.37 s of LAMMPS loop time respectively in those two tests.  The surrounding
  process wall time was higher because startup and dynamic loading are included.
- `build_kokkos_cuda/lmp -h`: successful; reports CUDA and Serial KOKKOS APIs.
- `rl_hpc/kokkos_cuda_smoke.py --steps 100`: successful overall; CPU versus
  one-GPU `gpu/aware off` thermo values matched at steps 0 and 100 with maximum
  absolute difference 0.0 for the recorded fields.
- Four-GPU `gpu/aware off` completed successfully.  Four-GPU `gpu/aware on`
  returned 139 and is recorded as the expected non-CUDA-aware-MPI diagnostic.
