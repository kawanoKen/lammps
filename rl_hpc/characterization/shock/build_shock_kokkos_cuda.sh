#!/usr/bin/env bash
set -euo pipefail

# Independent Shock/NEMD CUDA/KOKKOS build. Existing CPU and CUDA build
# directories are intentionally left untouched.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_shock_kokkos_cuda}"
BUILD_JOBS="${BUILD_JOBS:-16}"
CUDA_COMPILER="${CUDA_COMPILER:-$(command -v nvcc || true)}"
KOKKOS_ARCH="${KOKKOS_ARCH:-AMPERE86}"

if [[ -z "${CUDA_COMPILER}" ]]; then
  echo "nvcc was not found; set CUDA_COMPILER explicitly" >&2
  exit 1
fi

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
  -D BUILD_MPI=ON \
  -D BUILD_SHARED_LIBS=OFF \
  -D PKG_KOKKOS=ON \
  -D PKG_SHOCK=ON \
  -D PKG_EXTRA-PAIR=ON \
  -D PKG_EXTRA-COMPUTE=ON \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_SKX=ON \
  -D "Kokkos_ARCH_${KOKKOS_ARCH}=ON"

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
test -x "${BUILD_DIR}/lmp"

help="$(${BUILD_DIR}/lmp -h)"
grep -q 'SHOCK' <<<"${help}"
grep -q 'lj/cubic/kk' <<<"${help}"
grep -q 'wall/piston/kk' <<<"${help}"
grep -q 'momentum' <<<"${help}"
echo "READY ${BUILD_DIR}/lmp"
