#!/usr/bin/env bash
set -euo pipefail

# Independent CUDA/KOKKOS build.  This intentionally uses a different build
# directory from build_rl/, so the CPU/MPI reference build is left intact.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_kokkos_cuda}"
BUILD_JOBS="${BUILD_JOBS:-24}"
CUDA_COMPILER="${CUDA_COMPILER:-$(command -v nvcc || true)}"

if [[ -z "${CUDA_COMPILER}" ]]; then
  echo "nvcc was not found; set CUDA_COMPILER to the CUDA 13.4 nvcc path" >&2
  exit 1
fi

echo "Configuring independent KOKKOS/CUDA build in ${BUILD_DIR}"
cmake --fresh \
  -S "${ROOT_DIR}/cmake" \
  -B "${BUILD_DIR}" \
  -G 'Unix Makefiles' \
  -C "${ROOT_DIR}/cmake/presets/kokkos-cuda-nowrapper.cmake" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER="${CC:-gcc}" \
  -D CMAKE_CXX_COMPILER="${CXX:-g++}" \
  -D CMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
  -D CMAKE_CXX_STANDARD=20 \
  -D Kokkos_ARCH_SKX=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D BUILD_MPI=ON \
  -D BUILD_SHARED_LIBS=OFF \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D CMAKE_INSTALL_PREFIX="${BUILD_DIR}/install"

echo "Building with ${BUILD_JOBS} jobs"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"

test -x "${BUILD_DIR}/lmp"
echo "Built ${BUILD_DIR}/lmp"
