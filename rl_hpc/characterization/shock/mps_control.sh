#!/usr/bin/env bash
set -euo pipefail

PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/${USER}-lammps-mps-pipe}"
LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/${USER}-lammps-mps-log}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export CUDA_MPS_PIPE_DIRECTORY="${PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${LOG_DIR}"

case "${1:-status}" in
  start)
    mkdir -p "${PIPE_DIR}" "${LOG_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_LIST}" nvidia-cuda-mps-control -d
    echo "MPS started: GPUs=${GPU_LIST} pipe=${PIPE_DIR} log=${LOG_DIR}"
    ;;
  status)
    printf 'get_server_list\n' | nvidia-cuda-mps-control
    printf 'ps\n' | nvidia-cuda-mps-control
    ;;
  stop)
    printf 'quit\n' | nvidia-cuda-mps-control
    echo "MPS stopped: pipe=${PIPE_DIR}"
    ;;
  *)
    echo "usage: $0 {start|status|stop}" >&2
    exit 2
    ;;
esac
