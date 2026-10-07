// Small CUDA-only contender used for a controlled same-GPU co-run.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

__global__ void stream_kernel(float *a, const float *b, std::size_t n,
                              int repeats) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    float value = a[index];
    for (int repeat = 0; repeat < repeats; ++repeat) {
      value = value * 1.000001f + b[index];
    }
    a[index] = value;
  }
}

int main(int argc, char **argv) {
  const double duration = argc > 1 ? std::atof(argv[1]) : 30.0;
  // Arguments are duration_seconds, buffer_mb_per_array, flops_repeats,
  // sleep_us.  The last three are explicit intensity controls so that a
  // contention regime is reproducible and can be logged by the caller.
  const std::size_t buffer_mb = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 256;
  const int repeats = argc > 3 ? std::atoi(argv[3]) : 32;
  const int sleep_us = argc > 4 ? std::atoi(argv[4]) : 0;
  if (duration <= 0.0 || buffer_mb < 8 || repeats < 1 || sleep_us < 0) {
    std::fprintf(stderr, "usage: %s duration_seconds buffer_mb repeats sleep_us\n", argv[0]);
    return 1;
  }
  const std::size_t elements = buffer_mb * 1024ULL * 1024ULL / sizeof(float);
  float *a = nullptr;
  float *b = nullptr;
  if (cudaMalloc(&a, elements * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&b, elements * sizeof(float)) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    return 2;
  }
  cudaMemset(a, 0, elements * sizeof(float));
  cudaMemset(b, 1, elements * sizeof(float));
  const int blocks = static_cast<int>((elements + 255) / 256);
  int device = -1;
  cudaGetDevice(&device);
  std::printf("gpu_contender device=%d buffer_mb=%zu repeats=%d sleep_us=%d\n",
              device, buffer_mb, repeats, sleep_us);
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration<double>(duration);
  while (std::chrono::steady_clock::now() < deadline) {
    stream_kernel<<<blocks, 256>>>(a, b, elements, repeats);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::fprintf(stderr, "CUDA kernel failed\n");
      cudaFree(a);
      cudaFree(b);
      return 3;
    }
    if (sleep_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
  }
  cudaFree(a);
  cudaFree(b);
  return 0;
}
