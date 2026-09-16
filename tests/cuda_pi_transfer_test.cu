#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/pi_transfer.h"

namespace {

using cugo::core::CardMask;
using cugo::game::PiTransferResult;
using cugo::game::PiTransferSelection;
using cugo::game::PiTransferStatus;
using cugo::game::ScoreOptions;
using cugo::game::TurnState50;

struct TransferSample {
  CardMask initial0;
  CardMask initial1;
  CardMask selection;
  CardMask final0;
  CardMask final1;
  PiTransferResult result;
  std::uint8_t actor;
  std::uint8_t requested;
  std::uint8_t gukjin_as_pi;
  std::uint8_t mode;
};

__host__ __device__ CardMask select_first_n(CardMask cards,
                                             std::uint8_t count) {
  CardMask selected = 0;
  while (count-- != 0 && cards != 0) {
    const auto card = cugo::core::pop_first_card(cards);
    selected |= cugo::core::card_bit(card);
  }
  return selected;
}

__host__ __device__ void prepare_sample(std::uint32_t i,
                                        std::uint64_t master_seed,
                                        TurnState50& state,
                                        std::uint8_t& actor,
                                        std::uint8_t& requested,
                                        ScoreOptions& options,
                                        CardMask& selection,
                                        std::uint8_t& mode) {
  cugo::core::SplitMix64 rng{
      cugo::core::derive_seed(master_seed, static_cast<std::uint64_t>(i))};
  const CardMask victim = rng.next_u64() & cugo::core::kShinMatgoDeckMask;
  const CardMask thief =
      (rng.next_u64() & cugo::core::kShinMatgoDeckMask) & ~victim;

  actor = static_cast<std::uint8_t>((i >> 1u) & 1u);
  requested = static_cast<std::uint8_t>((i >> 2u) % 5u);
  options = ScoreOptions{(i & 1u) != 0};
  mode = static_cast<std::uint8_t>((i / 7u) % 3u);

  if (actor == 0) {
    state.captured0 = thief;
    state.captured1 = victim;
  } else {
    state.captured0 = victim;
    state.captured1 = thief;
  }

  const CardMask candidates = cugo::game::pi_card_mask(victim, options);
  const std::uint8_t available =
      static_cast<std::uint8_t>(cugo::core::card_count(candidates));
  selection = 0;

  if (mode == 1 && requested != 0 && available > requested) {
    selection = select_first_n(candidates, requested);
  } else if (mode == 2 && requested != 0) {
    if (available > requested) {
      selection = candidates;
    } else if (available != 0) {
      selection = cugo::core::card_bit(cugo::core::make_card(0, 0));
    }
  }
}

__global__ void evaluate_transfers(TransferSample* samples,
                                   std::uint32_t n,
                                   std::uint64_t master_seed) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  TurnState50 state{};
  std::uint8_t actor = 0;
  std::uint8_t requested = 0;
  ScoreOptions options{};
  CardMask selection = 0;
  std::uint8_t mode = 0;
  prepare_sample(i, master_seed, state, actor, requested, options, selection, mode);

  TransferSample sample{};
  sample.initial0 = state.captured0;
  sample.initial1 = state.captured1;
  sample.selection = selection;
  sample.actor = actor;
  sample.requested = requested;
  sample.gukjin_as_pi = options.gukjin_as_double_pi ? 1u : 0u;
  sample.mode = mode;
  sample.result = cugo::game::apply_pi_steal50(
      state, actor, requested, PiTransferSelection{selection}, options);
  sample.final0 = state.captured0;
  sample.final1 = state.captured1;
  samples[i] = sample;
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool same_result(const PiTransferResult& a, const PiTransferResult& b) {
  return a.status == b.status && a.requested_cards == b.requested_cards &&
         a.available_cards == b.available_cards &&
         a.transferred_cards == b.transferred_cards &&
         a.transferred_pi_units == b.transferred_pi_units &&
         a.transferred_mask == b.transferred_mask;
}

}  // namespace

int main() {
  constexpr std::uint32_t kSamples = 1u << 16;
  constexpr int kThreads = 256;
  constexpr std::uint64_t kSeed = 0x70695f7472616e73ULL;

  TransferSample* device = nullptr;
  if (!check_cuda(cudaMalloc(&device, sizeof(TransferSample) * kSamples),
                  "cudaMalloc")) {
    return 1;
  }

  evaluate_transfers<<<(kSamples + kThreads - 1) / kThreads, kThreads>>>(
      device, kSamples, kSeed);
  if (!check_cuda(cudaGetLastError(), "pi transfer kernel launch") ||
      !check_cuda(cudaDeviceSynchronize(), "pi transfer kernel synchronize")) {
    cudaFree(device);
    return 1;
  }

  std::vector<TransferSample> gpu(kSamples);
  if (!check_cuda(cudaMemcpy(gpu.data(),
                             device,
                             sizeof(TransferSample) * kSamples,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy")) {
    cudaFree(device);
    return 1;
  }
  cudaFree(device);

  std::uint32_t ok = 0;
  std::uint32_t selection_required = 0;
  std::uint32_t invalid_selection = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    TurnState50 state{};
    std::uint8_t actor = 0;
    std::uint8_t requested = 0;
    ScoreOptions options{};
    CardMask selection = 0;
    std::uint8_t mode = 0;
    prepare_sample(i, kSeed, state, actor, requested, options, selection, mode);

    const CardMask initial0 = state.captured0;
    const CardMask initial1 = state.captured1;
    const auto cpu = cugo::game::apply_pi_steal50(
        state, actor, requested, PiTransferSelection{selection}, options);

    const auto& sample = gpu[i];
    if (sample.initial0 != initial0 || sample.initial1 != initial1 ||
        sample.selection != selection || sample.actor != actor ||
        sample.requested != requested ||
        sample.gukjin_as_pi != (options.gukjin_as_double_pi ? 1u : 0u) ||
        sample.mode != mode || sample.final0 != state.captured0 ||
        sample.final1 != state.captured1 || !same_result(sample.result, cpu)) {
      std::cerr << "CPU/GPU pi-transfer mismatch at sample " << i << '\n';
      return 2;
    }

    switch (cpu.status) {
      case PiTransferStatus::kOk:
        ++ok;
        break;
      case PiTransferStatus::kSelectionRequired:
        ++selection_required;
        break;
      case PiTransferStatus::kInvalidSelection:
        ++invalid_selection;
        break;
    }
  }

  if (ok == 0 || selection_required == 0 || invalid_selection == 0) {
    std::cerr << "pi-transfer coverage missing: ok=" << ok
              << " selection_required=" << selection_required
              << " invalid_selection=" << invalid_selection << '\n';
    return 3;
  }

  std::cout << "cugo_cuda_pi_transfer_test: PASS (" << kSamples
            << " differential samples; ok=" << ok
            << " selection_required=" << selection_required
            << " invalid_selection=" << invalid_selection << ")\n";
  return 0;
}
