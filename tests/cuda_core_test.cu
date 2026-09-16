#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"

namespace {

struct Sample {
  std::uint64_t mask;
  std::uint64_t rng;
  std::uint8_t first;
  int count;
};

__global__ void evaluate_samples(Sample* samples, int n, std::uint64_t seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  using namespace cugo::core;
  SplitMix64 rng{seed + static_cast<std::uint64_t>(i)};
  const std::uint64_t bits = rng.next_u64() & kFullDeckMask;
  samples[i] = Sample{bits, rng.next_u64(), first_card(bits), card_count(bits)};
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

}  // namespace

int main() {
  using namespace cugo::core;
  constexpr int kSamples = 1 << 16;
  constexpr int kThreads = 256;
  constexpr std::uint64_t kSeed = 0x5eed1234cafebabeULL;

  Sample* device = nullptr;
  if (!check_cuda(cudaMalloc(&device, sizeof(Sample) * kSamples), "cudaMalloc")) {
    return 1;
  }

  evaluate_samples<<<(kSamples + kThreads - 1) / kThreads, kThreads>>>(device, kSamples, kSeed);
  if (!check_cuda(cudaGetLastError(), "kernel launch")) {
    cudaFree(device);
    return 1;
  }

  std::vector<Sample> gpu(kSamples);
  if (!check_cuda(cudaMemcpy(gpu.data(), device, sizeof(Sample) * kSamples, cudaMemcpyDeviceToHost),
                  "cudaMemcpy")) {
    cudaFree(device);
    return 1;
  }
  cudaFree(device);

  for (int i = 0; i < kSamples; ++i) {
    SplitMix64 rng{kSeed + static_cast<std::uint64_t>(i)};
    const std::uint64_t bits = rng.next_u64() & kFullDeckMask;
    const Sample cpu{bits, rng.next_u64(), first_card(bits), card_count(bits)};
    const Sample& actual = gpu[i];
    if (actual.mask != cpu.mask || actual.rng != cpu.rng || actual.first != cpu.first ||
        actual.count != cpu.count) {
      std::cerr << "CPU/GPU mismatch at sample " << i << '\n';
      return 2;
    }
  }

  std::cout << "cugo_cuda_core_test: PASS (" << kSamples << " differential samples)\n";
  return 0;
}
