#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_shock_char}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

if command -v ninja >/dev/null 2>&1; then
  generator=Ninja
else
  generator="Unix Makefiles"
fi

cmake --fresh -S "${ROOT_DIR}/cmake" -B "${BUILD_DIR}" -G "${generator}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=ON \
  -D PKG_OPENMP=ON \
  -D PKG_SHOCK=ON \
  -D CMAKE_CXX_FLAGS=-march=native
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
"${BUILD_DIR}/lmp" -h >/dev/null
echo "READY ${BUILD_DIR}/lmp"
