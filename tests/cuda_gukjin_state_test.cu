#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/game/pi_transfer.h"

namespace {

using cugo::core::CardMask;
using cugo::game::PersistentGukjinRole50;
using cugo::game::PiTransferResult;
using cugo::game::PiTransferSelection;
using cugo::game::TurnState50;

struct GukjinSample {
  CardMask final0;
  CardMask final1;
  CardMask transfer_mask;
  std::uint16_t packed_metadata;
  std::uint8_t before_animal_count;
  std::uint8_t before_pi_units;
  std::uint8_t after_owner0_role;
  std::uint8_t after_owner1_role;
  std::uint8_t transferred_pi_units;
  std::uint8_t transfer_status;
};

__host__ __device__ void prepare_state(std::uint32_t i,
                                       TurnState50& state,
                                       std::uint8_t& stealing_actor,
                                       CardMask& selection) {
  using namespace cugo::core;
  using namespace cugo::game;

  const std::uint8_t victim = static_cast<std::uint8_t>(i & 1u);
  stealing_actor = static_cast<std::uint8_t>(victim ^ 1u);
  const bool as_double_pi = (i & 2u) != 0;
  const CardMask victim_cards =
      card_bit(kGukjin) | card_bit(make_card(0, 2));

  if (victim == 0) state.captured0 = victim_cards;
  else state.captured1 = victim_cards;

  if (as_double_pi) {
    (void)set_persistent_gukjin_role50(
        state, victim, PersistentGukjinRole50::kDoublePi);
    selection = card_bit(kGukjin);
  } else {
    selection = card_bit(make_card(0, 2));
  }
}

__host__ __device__ GukjinSample evaluate_sample(std::uint32_t i) {
  using namespace cugo::game;

  TurnState50 state{};
  std::uint8_t stealing_actor = 0;
  CardMask selection = 0;
  prepare_state(i, state, stealing_actor, selection);

  const std::uint8_t victim = static_cast<std::uint8_t>(stealing_actor ^ 1u);
  const auto before = persistent_score_player50(state, victim);
  const PiTransferResult transfer = apply_pi_steal50(
      state, stealing_actor, 1, PiTransferSelection{selection});

  return GukjinSample{
      state.captured0,
      state.captured1,
      transfer.transferred_mask,
      state.ppuk_owner1_months,
      before.animal_count,
      before.pi_units,
      static_cast<std::uint8_t>(persistent_gukjin_role50(state, 0)),
      static_cast<std::uint8_t>(persistent_gukjin_role50(state, 1)),
      transfer.transferred_pi_units,
      static_cast<std::uint8_t>(transfer.status),
  };
}

__global__ void evaluate_kernel(GukjinSample* samples, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  samples[i] = evaluate_sample(i);
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return true;
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool same_sample(const GukjinSample& a, const GukjinSample& b) {
  return a.final0 == b.final0 && a.final1 == b.final1 &&
         a.transfer_mask == b.transfer_mask &&
         a.packed_metadata == b.packed_metadata &&
         a.before_animal_count == b.before_animal_count &&
         a.before_pi_units == b.before_pi_units &&
         a.after_owner0_role == b.after_owner0_role &&
         a.after_owner1_role == b.after_owner1_role &&
         a.transferred_pi_units == b.transferred_pi_units &&
         a.transfer_status == b.transfer_status;
}

}  // namespace

int main() {
  constexpr std::uint32_t kSamples = 1u << 16;
  constexpr int kThreads = 256;

  GukjinSample* device = nullptr;
  if (!check_cuda(cudaMalloc(&device, sizeof(GukjinSample) * kSamples),
                  "cudaMalloc")) {
    return 1;
  }

  evaluate_kernel<<<(kSamples + kThreads - 1) / kThreads, kThreads>>>(
      device, kSamples);
  if (!check_cuda(cudaGetLastError(), "gukjin kernel launch") ||
      !check_cuda(cudaDeviceSynchronize(), "gukjin kernel synchronize")) {
    cudaFree(device);
    return 1;
  }

  std::vector<GukjinSample> gpu(kSamples);
  if (!check_cuda(cudaMemcpy(gpu.data(), device,
                             sizeof(GukjinSample) * kSamples,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy")) {
    cudaFree(device);
    return 1;
  }
  cudaFree(device);

  std::uint32_t animal = 0;
  std::uint32_t double_pi = 0;
  std::uint32_t transferred_gukjin = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const GukjinSample cpu = evaluate_sample(i);
    if (!same_sample(gpu[i], cpu)) {
      std::cerr << "CPU/GPU Gukjin-state mismatch at sample " << i << '\n';
      return 2;
    }
    if ((i & 2u) != 0) {
      ++double_pi;
      if (cpu.transfer_mask == cugo::core::card_bit(cugo::game::kGukjin)) {
        ++transferred_gukjin;
      }
    } else {
      ++animal;
    }
  }

  if (animal == 0 || double_pi == 0 || transferred_gukjin != double_pi) {
    std::cerr << "Gukjin coverage missing animal=" << animal
              << " double_pi=" << double_pi
              << " transferred_gukjin=" << transferred_gukjin << '\n';
    return 3;
  }

  std::cout << "cugo_cuda_gukjin_state_test: PASS (" << kSamples
            << " persistent-role differential samples; animal=" << animal
            << " double_pi=" << double_pi
            << " transferred_gukjin=" << transferred_gukjin << ")\n";
  return 0;
}
