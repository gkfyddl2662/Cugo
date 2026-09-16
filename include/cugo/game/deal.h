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
inline constexpr int kBaseStockCards =
    core::kStandardCardCount - kPlayerCount * kInitialHandCards - kInitialFloorCards;
inline constexpr int kShinMatgoStockCards =
    core::kShinMatgoCardCount - kPlayerCount * kInitialHandCards - kInitialFloorCards;
inline constexpr int kInitialSampledCards =
    kPlayerCount * kInitialHandCards + kInitialFloorCards;

struct InitialDeal48 {
  CardMask hand0;
  CardMask hand1;
  CardMask floor;
  CardMask stock;
  std::uint64_t rng_state;
};

struct InitialDeal50 {
  CardMask hand0;
  CardMask hand1;
  CardMask floor;
  CardMask stock;
  CardMask captured0;
  CardMask captured1;
  std::uint64_t rng_state;
};

CUGO_HOST_DEVICE inline CardId draw_uniform_card(CardMask& remaining,
                                                 core::SplitMix64& rng) noexcept {
  const std::uint32_t count = static_cast<std::uint32_t>(core::card_count(remaining));
  if (count == 0u) return core::kInvalidCard;
  const std::uint32_t rank = core::uniform_bounded(rng, count);
  const CardId card = core::select_card_by_rank(remaining, rank);
  remaining &= ~core::card_bit(card);
  return card;
}

CUGO_HOST_DEVICE inline InitialDeal48 deal_base_48(std::uint64_t seed) noexcept {
  core::SplitMix64 rng{seed};
  CardMask remaining = core::kStandardDeckMask;
  CardMask hand0 = 0;
  CardMask hand1 = 0;
  CardMask floor = 0;
  for (int i = 0; i < kInitialHandCards; ++i) hand0 |= core::card_bit(draw_uniform_card(remaining, rng));
  for (int i = 0; i < kInitialHandCards; ++i) hand1 |= core::card_bit(draw_uniform_card(remaining, rng));
  for (int i = 0; i < kInitialFloorCards; ++i) floor |= core::card_bit(draw_uniform_card(remaining, rng));
  return InitialDeal48{hand0, hand1, floor, remaining, rng.state()};
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

CUGO_HOST_DEVICE inline bool is_valid_initial_deal(const InitialDeal48& deal) noexcept {
  if (core::card_count(deal.hand0) != kInitialHandCards ||
      core::card_count(deal.hand1) != kInitialHandCards ||
      core::card_count(deal.floor) != kInitialFloorCards ||
      core::card_count(deal.stock) != kBaseStockCards) return false;
  const CardMask overlap = (deal.hand0 & deal.hand1) | (deal.hand0 & deal.floor) |
                           (deal.hand0 & deal.stock) | (deal.hand1 & deal.floor) |
                           (deal.hand1 & deal.stock) | (deal.floor & deal.stock);
  if (overlap != 0) return false;
  return (deal.hand0 | deal.hand1 | deal.floor | deal.stock) == core::kStandardDeckMask;
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
