#!/usr/bin/env bash
set -euo pipefail

# Separate CPU characterization build.  The existing build_rl/ reference is
# deliberately not modified; KSPACE/MOLECULE are needed by the SPC/E workload.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_rl_char}"
BUILD_JOBS="${BUILD_JOBS:-24}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(command -v python3)}"

cmake --fresh \
  -S "${ROOT_DIR}/cmake" \
  -B "${BUILD_DIR}" \
  -G 'Unix Makefiles' \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_C_COMPILER="${CC:-gcc}" \
  -D CMAKE_CXX_COMPILER="${CXX:-g++}" \
  -D CMAKE_CXX_FLAGS=-march=native \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=ON \
  -D PKG_OPENMP=ON \
  -D PKG_PYTHON=ON \
  -D PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
  -D PKG_KSPACE=ON \
  -D PKG_MOLECULE=ON \
  -D PKG_EXTRA-MOLECULE=ON \
  -D PKG_RIGID=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D CMAKE_INSTALL_PREFIX="${BUILD_DIR}/install"

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
test -x "${BUILD_DIR}/lmp"
test -f "${BUILD_DIR}/liblammps.so" || test -L "${BUILD_DIR}/liblammps.so"
echo "Built ${BUILD_DIR}/lmp"
