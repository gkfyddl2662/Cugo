#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cugo/game/policy50.h"

namespace {

using namespace cugo::game;

struct Probe {
  PolicyPacket50 packet;
  std::uint8_t valid;
};

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(1);
  }
}

__host__ __device__ Probe make_probe(std::uint32_t index) {
  TerminalGameState50 state = make_terminal_game_state50(
      deal_shin_matgo_50(0x43554441504f4c49ULL + index),
      static_cast<std::uint8_t>(index & 1u));
  const std::uint16_t target = static_cast<std::uint16_t>((index >> 1u) % 24u);
  std::uint8_t valid = 1;

  for (std::uint16_t step = 0; step < target; ++step) {
    if (terminal50_is_finished(state)) break;
    if (!terminal50_has_chongtong_choice(state) &&
        !game50_has_pending_decision(state.special.game) &&
        rollout_hands_exhausted50(state)) break;
    const CanonicalAction50 canonical = canonical_action50(state);
    if (canonical.status != ActionStatus50::kOk) {
      valid = 0;
      break;
    }
    if (apply_action50(state, canonical.action).status != ActionStatus50::kOk) {
      valid = 0;
      break;
    }
  }

  if (!is_valid_terminal_game_state50(state)) valid = 0;
  const PolicyPacket50 packet = make_policy_packet50(state);
  if ((packet.legal_actions.hi & (std::uint64_t{1} << 63u)) != 0) valid = 0;
  return Probe{packet, valid};
}

__global__ void policy_probe_kernel(Probe* out, std::uint32_t n) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = make_probe(index);
}

}  // namespace

int main() {
  constexpr std::uint32_t kSamples = 65536;
  std::vector<Probe> actual(kSamples);
  Probe* device = nullptr;
  cuda_check(cudaMalloc(&device, sizeof(Probe) * kSamples), "cudaMalloc");

  constexpr std::uint32_t kThreads = 256;
  const std::uint32_t blocks = (kSamples + kThreads - 1u) / kThreads;
  policy_probe_kernel<<<blocks, kThreads>>>(device, kSamples);
  cuda_check(cudaGetLastError(), "policy_probe_kernel launch");
  cuda_check(cudaDeviceSynchronize(), "policy_probe_kernel sync");
  cuda_check(cudaMemcpy(actual.data(), device, sizeof(Probe) * kSamples,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy");
  cuda_check(cudaFree(device), "cudaFree");

  std::uint64_t primary = 0;
  std::uint64_t go_stop = 0;
  std::uint64_t chongtong = 0;
  std::uint64_t none = 0;
  std::uint64_t legal_total = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const Probe expected = make_probe(i);
    if (expected.valid == 0 || actual[i].valid == 0 ||
        !equal_policy_packet50(expected.packet, actual[i].packet)) {
      std::cerr << "cugo_cuda_policy50_test: mismatch at sample " << i << '\n';
      return 1;
    }

    legal_total += policy_action_count50(actual[i].packet.legal_actions);
    switch (static_cast<PolicyDecision50>(actual[i].packet.observation.decision_kind)) {
      case PolicyDecision50::kPrimary: ++primary; break;
      case PolicyDecision50::kGoStop: ++go_stop; break;
      case PolicyDecision50::kChongtong: ++chongtong; break;
      default: ++none; break;
    }
  }

  std::cout << "cugo_cuda_policy50_test: PASS (" << kSamples
            << " policy packets; primary=" << primary
            << " go_stop=" << go_stop
            << " chongtong=" << chongtong
            << " none=" << none
            << " avg_legal="
            << static_cast<double>(legal_total) / static_cast<double>(kSamples)
            << " packet_bytes=" << sizeof(PolicyPacket50) << ")\n";
  return 0;
}
