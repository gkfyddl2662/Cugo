#pragma once

#include <cstdint>

#include "cugo/core/card.h"
#include "cugo/game/state.h"

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::game {

using core::CardId;
using core::CardMask;

enum class BonusPlayStatus : std::uint8_t {
  kOk = 0,
  kNotBonus = 1,
  kCardNotInHand = 2,
  kStockEmpty = 3,
};

struct BonusPlayResult {
  BonusPlayStatus status;
  CardId replacement;
  std::uint8_t pi_steal_count;
};

struct BonusFlipResult {
  CardId standard_card;
  CardMask pending_bonus_mask;
};

CUGO_HOST_DEVICE inline BonusPlayResult play_hand_bonus(
    CardMask& hand,
    CardMask& stock,
    CardMask& captured,
    std::uint64_t& rng_state,
    CardId bonus) noexcept {
  if (!core::is_bonus_card(bonus)) {
    return BonusPlayResult{BonusPlayStatus::kNotBonus, core::kInvalidCard, 0};
  }
  const CardMask bit = core::card_bit(bonus);
  if ((hand & bit) == 0) {
    return BonusPlayResult{BonusPlayStatus::kCardNotInHand, core::kInvalidCard, 0};
  }
  if (stock == 0) {
    return BonusPlayResult{BonusPlayStatus::kStockEmpty, core::kInvalidCard, 0};
  }

  CardMask next_stock = stock;
  std::uint64_t next_rng = rng_state;
  const CardId replacement = draw_stock_card(next_stock, next_rng);
  if (!core::is_physical_card(replacement)) {
    return BonusPlayResult{BonusPlayStatus::kStockEmpty, core::kInvalidCard, 0};
  }

  hand = (hand & ~bit) | core::card_bit(replacement);
  captured |= bit;
  stock = next_stock;
  rng_state = next_rng;
  return BonusPlayResult{BonusPlayStatus::kOk, replacement, 1};
}

CUGO_HOST_DEVICE inline BonusFlipResult draw_stock_with_bonus_chain(
    CardMask& stock,
    std::uint64_t& rng_state) noexcept {
  CardMask bonuses = 0;
  while (stock != 0) {
    const CardId card = draw_stock_card(stock, rng_state);
    if (core::is_bonus_card(card)) {
      bonuses |= core::card_bit(card);
      continue;
    }
    return BonusFlipResult{card, bonuses};
  }
  return BonusFlipResult{core::kInvalidCard, bonuses};
}

}  // namespace cugo::game

#undef CUGO_HOST_DEVICE
