#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "cugo/game/torch50.h"

namespace {

using namespace cugo::game;

struct DeviceRollout50 {
  RolloutDigest50 digest;
  std::uint16_t choice_actions;
  std::uint16_t resolve_choices;
  std::uint16_t pi_choices;
};

__host__ __device__ std::uint64_t rollout_seed(std::uint32_t i) {
  return 0x5be0cd19137e2179ull +
         static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull;
}

__host__ __device__ std::uint64_t packet_seed(std::uint32_t i) {
  return 0xcbbb9d5dc1059ed8ull +
         static_cast<std::uint64_t>(i) * 0x94d049bb133111ebull;
}

__global__ void rollout_kernel(DeviceRollout50* out, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const TorchRollout50 result = rollout_canonical_torch50(
      rollout_seed(i), static_cast<std::uint8_t>(i & 1u));
  out[i] = DeviceRollout50{rollout_digest50(result.result),
                            result.choice_actions,
                            result.resolve_choices,
                            result.pi_choices};
}

__global__ void packet_kernel(float* features,
                              TorchActionMask50* masks,
                              std::uint8_t* decisions,
                              std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  TorchEnv50 env = make_torch_env50(
      packet_seed(i), static_cast<std::uint8_t>(i & 1u));
  const std::uint16_t advance = static_cast<std::uint16_t>((i >> 1u) % 23u);
  for (std::uint16_t step = 0; step < advance && env.done == 0; ++step) {
    const std::uint16_t action = canonical_torch_action50(env);
    if (action == kInvalidTorch50Action) break;
    const TorchStepResult50 result = torch_step50(env, action);
    if (result.status != ActionStatus50::kOk) break;
  }

  const TorchPacket50 packet = make_torch_packet50(env);
  encode_torch_packet50(packet,
                        features + static_cast<std::size_t>(i) * kTorch50FeatureCount);
  masks[i] = packet.legal_actions;
  decisions[i] = packet.decision_kind;
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(1);
  }
}

void validate_packets() {
  constexpr std::uint32_t kSamples = 4096;
  const std::size_t feature_count =
      static_cast<std::size_t>(kSamples) * kTorch50FeatureCount;

  float* d_features = nullptr;
  TorchActionMask50* d_masks = nullptr;
  std::uint8_t* d_decisions = nullptr;
  check_cuda(cudaMalloc(&d_features, sizeof(float) * feature_count),
             "cudaMalloc features");
  check_cuda(cudaMalloc(&d_masks, sizeof(TorchActionMask50) * kSamples),
             "cudaMalloc masks");
  check_cuda(cudaMalloc(&d_decisions, sizeof(std::uint8_t) * kSamples),
             "cudaMalloc decisions");

  constexpr int kThreads = 128;
  const int blocks = static_cast<int>((kSamples + kThreads - 1) / kThreads);
  packet_kernel<<<blocks, kThreads>>>(d_features, d_masks, d_decisions, kSamples);
  check_cuda(cudaGetLastError(), "packet_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "packet_kernel sync");

  std::unique_ptr<float[]> features(new float[feature_count]);
  std::unique_ptr<TorchActionMask50[]> masks(new TorchActionMask50[kSamples]);
  std::unique_ptr<std::uint8_t[]> decisions(new std::uint8_t[kSamples]);
  check_cuda(cudaMemcpy(features.get(), d_features, sizeof(float) * feature_count,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy features");
  check_cuda(cudaMemcpy(masks.get(), d_masks,
                        sizeof(TorchActionMask50) * kSamples,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy masks");
  check_cuda(cudaMemcpy(decisions.get(), d_decisions,
                        sizeof(std::uint8_t) * kSamples,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy decisions");

  float expected[kTorch50FeatureCount];
  for (std::uint32_t i = 0; i < kSamples; ++i) {
    TorchEnv50 env = make_torch_env50(
        packet_seed(i), static_cast<std::uint8_t>(i & 1u));
    const std::uint16_t advance = static_cast<std::uint16_t>((i >> 1u) % 23u);
    for (std::uint16_t step = 0; step < advance && env.done == 0; ++step) {
      const std::uint16_t action = canonical_torch_action50(env);
      if (action == kInvalidTorch50Action) break;
      const TorchStepResult50 result = torch_step50(env, action);
      if (result.status != ActionStatus50::kOk) break;
    }
    const TorchPacket50 packet = make_torch_packet50(env);
    encode_torch_packet50(packet, expected);

    if (masks[i].word0 != packet.legal_actions.word0 ||
        masks[i].word1 != packet.legal_actions.word1 ||
        masks[i].word2 != packet.legal_actions.word2 ||
        decisions[i] != packet.decision_kind) {
      std::cerr << "cugo_cuda_torch50_test: FAIL packet metadata at " << i << '\n';
      std::exit(1);
    }

    const float* actual =
        features.get() + static_cast<std::size_t>(i) * kTorch50FeatureCount;
    for (std::uint16_t j = 0; j < kTorch50FeatureCount; ++j) {
      if (std::fabs(actual[j] - expected[j]) > 1.0e-6f) {
        std::cerr << "cugo_cuda_torch50_test: FAIL feature at sample " << i
                  << " index " << j << " gpu=" << actual[j]
                  << " cpu=" << expected[j] << '\n';
        std::exit(1);
      }
    }
  }

  check_cuda(cudaFree(d_features), "cudaFree features");
  check_cuda(cudaFree(d_masks), "cudaFree masks");
  check_cuda(cudaFree(d_decisions), "cudaFree decisions");
}

}  // namespace

int main() {
  validate_packets();

  constexpr std::uint32_t kSamples = 65536;
  DeviceRollout50* device = nullptr;
  check_cuda(cudaMalloc(&device, sizeof(DeviceRollout50) * kSamples),
             "cudaMalloc rollout");

  constexpr int kThreads = 128;
  const int blocks = static_cast<int>((kSamples + kThreads - 1) / kThreads);
  rollout_kernel<<<blocks, kThreads>>>(device, kSamples);
  check_cuda(cudaGetLastError(), "rollout_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "rollout_kernel sync");

  std::unique_ptr<DeviceRollout50[]> host(new DeviceRollout50[kSamples]);
  check_cuda(cudaMemcpy(host.get(), device,
                        sizeof(DeviceRollout50) * kSamples,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy rollout");

  std::uint32_t terminal = 0;
  std::uint32_t nagari = 0;
  std::uint64_t choice_actions = 0;
  std::uint64_t resolve_choices = 0;
  std::uint64_t pi_choices = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const RolloutDigest50 expected = rollout_digest50(rollout_canonical50(
        rollout_seed(i), static_cast<std::uint8_t>(i & 1u)));
    if (!equal_rollout_digest50(host[i].digest, expected)) {
      std::cerr << "cugo_cuda_torch50_test: FAIL rollout digest at " << i << '\n';
      cudaFree(device);
      return 1;
    }

    const auto end = static_cast<RolloutEnd50>(host[i].digest.end);
    if (end == RolloutEnd50::kTerminal) ++terminal;
    else if (end == RolloutEnd50::kNagari) ++nagari;
    else {
      std::cerr << "cugo_cuda_torch50_test: FAIL non-terminal sample " << i << '\n';
      cudaFree(device);
      return 1;
    }
    choice_actions += host[i].choice_actions;
    resolve_choices += host[i].resolve_choices;
    pi_choices += host[i].pi_choices;
  }

  check_cuda(cudaFree(device), "cudaFree rollout");

  if (terminal + nagari != kSamples || choice_actions == 0 ||
      resolve_choices == 0 || pi_choices == 0) {
    std::cerr << "cugo_cuda_torch50_test: FAIL coverage"
              << " terminal=" << terminal
              << " nagari=" << nagari
              << " choice_actions=" << choice_actions
              << " resolve_choices=" << resolve_choices
              << " pi_choices=" << pi_choices << '\n';
    return 1;
  }

  std::cout << "cugo_cuda_torch50_test: PASS (" << kSamples
            << " rollouts + 4096 packet differential samples; terminal=" << terminal
            << " nagari=" << nagari
            << " choice_actions=" << choice_actions
            << " resolve_choices=" << resolve_choices
            << " pi_choices=" << pi_choices
            << " features=" << kTorch50FeatureCount
            << " actions=" << kTorch50ActionCount << ")\n";
  return 0;
}
