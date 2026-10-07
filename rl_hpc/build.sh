#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_rl}"
BUILD_JOBS="${BUILD_JOBS:-24}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(command -v python3)}"

if command -v ninja >/dev/null 2>&1; then
  GENERATOR="Ninja"
else
  GENERATOR="Unix Makefiles"
fi

echo "Configuring LAMMPS in ${BUILD_DIR} using ${GENERATOR}"
cmake --fresh \
  -S "${ROOT_DIR}/cmake" \
  -B "${BUILD_DIR}" \
  -G "${GENERATOR}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER="${CC:-gcc}" \
  -D CMAKE_CXX_COMPILER="${CXX:-g++}" \
  -D CMAKE_CXX_FLAGS=-march=native \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=ON \
  -D PKG_OPENMP=ON \
  -D PKG_PYTHON=ON \
  -D PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
  -D BUILD_SHARED_LIBS=ON \
  -D CMAKE_INSTALL_PREFIX="${BUILD_DIR}/install"

echo "Building with ${BUILD_JOBS} jobs"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"

test -x "${BUILD_DIR}/lmp"
test -f "${BUILD_DIR}/liblammps.so" || test -L "${BUILD_DIR}/liblammps.so"
echo "Built ${BUILD_DIR}/lmp"
