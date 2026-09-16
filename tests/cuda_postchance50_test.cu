#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cugo/game/postchance50.h"

namespace {

using namespace cugo::game;

struct DeviceSample50 {
  RolloutDigest50 digest;
  std::uint16_t resolve_choices;
  std::uint16_t pi_choices;
};

__host__ __device__ std::uint64_t sample_seed(std::uint32_t i) {
  return 0xa54ff53a5f1d36f1ull +
         static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull;
}

__global__ void postchance_kernel(DeviceSample50* out, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const PostChanceRollout50 result = rollout_canonical_postchance50(
      sample_seed(i), static_cast<std::uint8_t>(i & 1u));
  out[i] = DeviceSample50{rollout_digest50(result.result),
                          result.resolve_choices,
                          result.pi_choices};
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(1);
  }
}

}  // namespace

int main() {
  constexpr std::uint32_t kSamples = 65536;
  DeviceSample50* device = nullptr;
  check_cuda(cudaMalloc(&device, sizeof(DeviceSample50) * kSamples),
             "cudaMalloc");

  constexpr int kThreads = 128;
  const int blocks = static_cast<int>((kSamples + kThreads - 1) / kThreads);
  postchance_kernel<<<blocks, kThreads>>>(device, kSamples);
  check_cuda(cudaGetLastError(), "postchance_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "postchance_kernel sync");

  auto* host = new DeviceSample50[kSamples];
  check_cuda(cudaMemcpy(host, device, sizeof(DeviceSample50) * kSamples,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy");

  std::uint32_t terminal = 0;
  std::uint32_t nagari = 0;
  std::uint64_t resolve_choices = 0;
  std::uint64_t pi_choices = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const RolloutDigest50 expected = rollout_digest50(rollout_canonical50(
        sample_seed(i), static_cast<std::uint8_t>(i & 1u)));
    if (!equal_rollout_digest50(host[i].digest, expected)) {
      std::cerr << "cugo_cuda_postchance50_test: FAIL at sample " << i << '\n';
      delete[] host;
      cudaFree(device);
      return 1;
    }

    const auto end = static_cast<RolloutEnd50>(host[i].digest.end);
    if (end == RolloutEnd50::kTerminal) ++terminal;
    else if (end == RolloutEnd50::kNagari) ++nagari;
    else {
      std::cerr << "cugo_cuda_postchance50_test: FAIL non-terminal sample "
                << i << '\n';
      delete[] host;
      cudaFree(device);
      return 1;
    }
    resolve_choices += host[i].resolve_choices;
    pi_choices += host[i].pi_choices;
  }

  delete[] host;
  check_cuda(cudaFree(device), "cudaFree");

  if (terminal + nagari != kSamples || resolve_choices == 0 || pi_choices == 0) {
    std::cerr << "cugo_cuda_postchance50_test: FAIL coverage"
              << " terminal=" << terminal
              << " nagari=" << nagari
              << " resolve_choices=" << resolve_choices
              << " pi_choices=" << pi_choices << '\n';
    return 1;
  }

  std::cout << "cugo_cuda_postchance50_test: PASS (" << kSamples
            << " explicit-decision rollouts; terminal=" << terminal
            << " nagari=" << nagari
            << " resolve_choices=" << resolve_choices
            << " pi_choices=" << pi_choices
            << " packet_bytes=" << sizeof(PostChancePacket50) << ")\n";
  return 0;
}
