#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/bonus.h"
#include "cugo/game/deal.h"

namespace {

struct BonusSnapshot {
  std::uint64_t hand0;
  std::uint64_t hand1;
  std::uint64_t floor;
  std::uint64_t stock;
  std::uint64_t captured0;
  std::uint64_t captured1;
  std::uint64_t rng_state;
  std::uint64_t initial_collected;
  std::uint64_t pending_stock_bonuses;
  std::uint8_t bonus_status;
  std::uint8_t replacement;
  std::uint8_t pi_steal_count;
  std::uint8_t standard_draw;
};

__host__ __device__ BonusSnapshot evaluate_one(std::uint64_t master_seed,
                                               std::uint64_t game) {
  using namespace cugo::core;
  using namespace cugo::game;

  const std::uint64_t seed = derive_seed(master_seed, game);
  auto deal = deal_shin_matgo_50(seed);
  const std::uint8_t actor = static_cast<std::uint8_t>(game & 1u);
  const CardMask collected = collect_initial_floor_bonuses(deal, actor);

  CardMask& hand = actor == 0 ? deal.hand0 : deal.hand1;
  CardMask& captured = actor == 0 ? deal.captured0 : deal.captured1;
  const CardId bonus = first_card(hand & kBonusCardMask);
  const auto played = play_hand_bonus(
      hand, deal.stock, captured, deal.rng_state, bonus);
  const auto flipped = draw_stock_with_bonus_chain(deal.stock, deal.rng_state);

  return BonusSnapshot{
      deal.hand0,
      deal.hand1,
      deal.floor,
      deal.stock,
      deal.captured0,
      deal.captured1,
      deal.rng_state,
      collected,
      flipped.pending_bonus_mask,
      static_cast<std::uint8_t>(played.status),
      played.replacement,
      played.pi_steal_count,
      flipped.standard_card,
  };
}

__global__ void evaluate_bonus_samples(BonusSnapshot* samples,
                                       int n,
                                       std::uint64_t master_seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  samples[i] = evaluate_one(master_seed, static_cast<std::uint64_t>(i));
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return true;
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool same_snapshot(const BonusSnapshot& a, const BonusSnapshot& b) {
  return a.hand0 == b.hand0 && a.hand1 == b.hand1 &&
         a.floor == b.floor && a.stock == b.stock &&
         a.captured0 == b.captured0 && a.captured1 == b.captured1 &&
         a.rng_state == b.rng_state &&
         a.initial_collected == b.initial_collected &&
         a.pending_stock_bonuses == b.pending_stock_bonuses &&
         a.bonus_status == b.bonus_status &&
         a.replacement == b.replacement &&
         a.pi_steal_count == b.pi_steal_count &&
         a.standard_draw == b.standard_draw;
}

}  // namespace

int main() {
  constexpr int kSamples = 1 << 16;
  constexpr int kThreads = 256;
  constexpr std::uint64_t kMasterSeed = 0x626f6e75735f6770ULL;

  BonusSnapshot* device = nullptr;
  if (!check_cuda(cudaMalloc(&device, sizeof(BonusSnapshot) * kSamples),
                  "cudaMalloc")) {
    return 1;
  }

  evaluate_bonus_samples<<<(kSamples + kThreads - 1) / kThreads, kThreads>>>(
      device, kSamples, kMasterSeed);
  if (!check_cuda(cudaGetLastError(), "bonus kernel launch") ||
      !check_cuda(cudaDeviceSynchronize(), "bonus kernel synchronize")) {
    cudaFree(device);
    return 1;
  }

  std::vector<BonusSnapshot> gpu(kSamples);
  if (!check_cuda(cudaMemcpy(gpu.data(), device,
                             sizeof(BonusSnapshot) * kSamples,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy")) {
    cudaFree(device);
    return 1;
  }
  cudaFree(device);

  int initial_bonus = 0;
  int hand_bonus = 0;
  int stock_bonus = 0;
  for (int i = 0; i < kSamples; ++i) {
    const BonusSnapshot cpu =
        evaluate_one(kMasterSeed, static_cast<std::uint64_t>(i));
    if (!same_snapshot(gpu[i], cpu)) {
      std::cerr << "CPU/GPU bonus mismatch at sample " << i << '\n';
      return 2;
    }
    initial_bonus += cpu.initial_collected != 0;
    hand_bonus += cpu.bonus_status ==
                  static_cast<std::uint8_t>(cugo::game::BonusPlayStatus::kOk);
    stock_bonus += cpu.pending_stock_bonuses != 0;
  }

  if (initial_bonus == 0 || hand_bonus == 0 || stock_bonus == 0) {
    std::cerr << "Bonus coverage missing: initial=" << initial_bonus
              << " hand=" << hand_bonus << " stock=" << stock_bonus << '\n';
    return 3;
  }

  std::cout << "cugo_cuda_bonus_test: PASS (" << kSamples
            << " differential samples; initial_bonus=" << initial_bonus
            << " hand_bonus=" << hand_bonus
            << " stock_bonus=" << stock_bonus << ")\n";
  return 0;
}
