#pragma once

#include <cstdint>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::game {

using core::CardId;
using core::CardMask;

inline constexpr int kPlayerCount = 2;
inline constexpr int kInitialHandCards = 10;
inline constexpr int kInitialFloorCards = 8;
inline constexpr int kShinMatgoStockCards =
    core::kShinMatgoCardCount - kPlayerCount * kInitialHandCards - kInitialFloorCards;
inline constexpr int kInitialSampledCards =
    kPlayerCount * kInitialHandCards + kInitialFloorCards;

struct InitialDeal50 {
  CardMask hand0;
  CardMask hand1;
  CardMask floor;
  CardMask stock;
  CardMask captured0;
  CardMask captured1;
  std::uint64_t rng_state;
};

#if !defined(__CUDA_ARCH__)
// Host-only replay hook: when set, stock draws take these cards in order instead of
// sampling (used to replay games recorded from another implementation). A forced card
// missing from the remaining stock sets `mismatch` and falls back to sampling.
struct ForcedDraws50 {
  const CardId* cards;
  int count;
  int next;
  bool mismatch;
};
inline thread_local ForcedDraws50* g_forced_draws50 = nullptr;
#endif

CUGO_HOST_DEVICE inline CardId draw_uniform_card(CardMask& remaining,
                                                 core::SplitMix64& rng) noexcept {
  const std::uint32_t count = static_cast<std::uint32_t>(core::card_count(remaining));
  if (count == 0u) return core::kInvalidCard;
#if !defined(__CUDA_ARCH__)
  if (ForcedDraws50* forced = g_forced_draws50; forced != nullptr && forced->next < forced->count) {
    const CardId card = forced->cards[forced->next++];
    if ((remaining & core::card_bit(card)) != 0) {
      remaining &= ~core::card_bit(card);
      return card;
    }
    forced->mismatch = true;
  }
#endif
  const std::uint32_t rank = core::uniform_bounded(rng, count);
  const CardId card = core::select_card_by_rank(remaining, rank);
  remaining &= ~core::card_bit(card);
  return card;
}

CUGO_HOST_DEVICE inline CardId draw_stock_card(CardMask& stock,
                                               std::uint64_t& rng_state) noexcept {
  if (stock == 0) {
    return core::kInvalidCard;
  }

  core::SplitMix64 rng{rng_state};
  const CardId card = draw_uniform_card(stock, rng);
  rng_state = rng.state();
  return card;
}

CUGO_HOST_DEVICE inline InitialDeal50 deal_shin_matgo_50(std::uint64_t seed) noexcept {
  core::SplitMix64 rng{seed};
  CardMask remaining = core::kShinMatgoDeckMask;
  CardMask hand0 = 0;
  CardMask hand1 = 0;
  CardMask floor = 0;
  for (int i = 0; i < kInitialHandCards; ++i) hand0 |= core::card_bit(draw_uniform_card(remaining, rng));
  for (int i = 0; i < kInitialHandCards; ++i) hand1 |= core::card_bit(draw_uniform_card(remaining, rng));
  for (int i = 0; i < kInitialFloorCards; ++i) floor |= core::card_bit(draw_uniform_card(remaining, rng));
  return InitialDeal50{hand0, hand1, floor, remaining, 0, 0, rng.state()};
}

CUGO_HOST_DEVICE inline CardMask collect_initial_floor_bonuses(
    InitialDeal50& deal, std::uint8_t first_player) noexcept {
  const CardMask bonuses = deal.floor & core::kBonusCardMask;
  deal.floor &= ~bonuses;
  if ((first_player & 1u) == 0) deal.captured0 |= bonuses;
  else deal.captured1 |= bonuses;
  return bonuses;
}

CUGO_HOST_DEVICE inline bool is_valid_initial_deal(const InitialDeal50& deal) noexcept {
  const CardMask captured = deal.captured0 | deal.captured1;
  if (core::card_count(deal.hand0) != kInitialHandCards ||
      core::card_count(deal.hand1) != kInitialHandCards ||
      core::card_count(deal.stock) != kShinMatgoStockCards ||
      core::card_count(deal.floor) + core::card_count(captured) != kInitialFloorCards ||
      (captured & ~core::kBonusCardMask) != 0 ||
      (deal.captured0 & deal.captured1) != 0) return false;
  CardMask seen = 0;
  const CardMask parts[] = {deal.hand0, deal.hand1, deal.floor, deal.stock,
                            deal.captured0, deal.captured1};
  for (CardMask part : parts) {
    if ((seen & part) != 0) return false;
    seen |= part;
  }
  return seen == core::kShinMatgoDeckMask;
}

}  // namespace cugo::game

#undef CUGO_HOST_DEVICE
