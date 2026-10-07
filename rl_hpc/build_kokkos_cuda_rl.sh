#!/usr/bin/env bash
set -euo pipefail

# Dedicated shared CUDA/KOKKOS library for persistent Python-controlled RL
# episodes. Existing CPU and executable-only CUDA builds are untouched.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_kokkos_cuda_rl}"
cmake --fresh -S "${ROOT_DIR}/cmake" -B "${BUILD_DIR}" -G 'Unix Makefiles' \
  -C "${ROOT_DIR}/cmake/presets/kokkos-cuda-nowrapper.cmake" \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER="${CC:-gcc}" \
  -D CMAKE_CXX_COMPILER="${CXX:-g++}" -D CMAKE_CUDA_COMPILER="${CUDA_COMPILER:-$(command -v nvcc)}" \
  -D CMAKE_CXX_STANDARD=20 -D Kokkos_ARCH_SKX=ON -D Kokkos_ARCH_AMPERE86=ON \
  -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON -D PKG_KOKKOS=ON -D PKG_PYTHON=ON \
  -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_CUDA=ON
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-24}"
test -f "${BUILD_DIR}/liblammps.so"
