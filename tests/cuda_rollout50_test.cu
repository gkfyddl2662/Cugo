#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cugo/game/rollout50.h"

namespace {

using namespace cugo::game;

__host__ __device__ std::uint64_t sample_seed(std::uint32_t i) {
  return 0x243f6a8885a308d3ull +
         static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull;
}

__host__ __device__ RolloutDigest50 run_sample(std::uint32_t i) {
  const RolloutResult50 result = rollout_canonical50(
      sample_seed(i), static_cast<std::uint8_t>(i & 1u));
  return rollout_digest50(result);
}

__global__ void rollout_kernel(RolloutDigest50* out, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = run_sample(i);
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
  RolloutDigest50* device = nullptr;
  check_cuda(cudaMalloc(&device, sizeof(RolloutDigest50) * kSamples),
             "cudaMalloc");

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kSamples + kThreads - 1) / kThreads);
  rollout_kernel<<<blocks, kThreads>>>(device, kSamples);
  check_cuda(cudaGetLastError(), "rollout_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "rollout_kernel sync");

  auto* host = new RolloutDigest50[kSamples];
  check_cuda(cudaMemcpy(host, device, sizeof(RolloutDigest50) * kSamples,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy");

  std::uint32_t terminal = 0;
  std::uint32_t nagari = 0;
  std::uint32_t stalled = 0;
  std::uint32_t action_error = 0;
  std::uint32_t max_actions = 0;
  std::uint64_t bonus_actions = 0;
  std::uint64_t special_actions = 0;
  std::uint64_t go_actions = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const RolloutDigest50 expected = run_sample(i);
    if (!equal_rollout_digest50(host[i], expected)) {
      std::cerr << "cugo_cuda_rollout50_test: FAIL at sample " << i << '\n';
      delete[] host;
      cudaFree(device);
      return 1;
    }

    const auto end = static_cast<RolloutEnd50>(host[i].end);
    if (end == RolloutEnd50::kTerminal) ++terminal;
    else if (end == RolloutEnd50::kNagari) ++nagari;
    else if (end == RolloutEnd50::kStalled) ++stalled;
    else if (end == RolloutEnd50::kActionError) ++action_error;
    else if (end == RolloutEnd50::kMaxActions) ++max_actions;
    bonus_actions += host[i].bonus_actions;
    special_actions += host[i].special_actions;
    go_actions += host[i].go_actions;
  }

  delete[] host;
  check_cuda(cudaFree(device), "cudaFree");

  if (stalled != 0 || action_error != 0 || max_actions != 0 ||
      terminal + nagari != kSamples || bonus_actions == 0 ||
      special_actions == 0 || go_actions == 0) {
    std::cerr << "cugo_cuda_rollout50_test: FAIL coverage/termination"
              << " terminal=" << terminal
              << " nagari=" << nagari
              << " stalled=" << stalled
              << " action_error=" << action_error
              << " max_actions=" << max_actions << '\n';
    return 1;
  }

  std::cout << "cugo_cuda_rollout50_test: PASS (" << kSamples
            << " full-game differential samples; terminal=" << terminal
            << " nagari=" << nagari
            << " bonus_actions=" << bonus_actions
            << " special_actions=" << special_actions
            << " go_actions=" << go_actions << ")\n";
  return 0;
}
