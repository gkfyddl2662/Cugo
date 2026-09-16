#pragma once

#include <bit>
#include <cstdint>

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::core {

using CardId = std::uint8_t;
using CardMask = std::uint64_t;

inline constexpr int kMonthCount = 12;
inline constexpr int kCardsPerMonth = 4;
inline constexpr int kCardCount = kMonthCount * kCardsPerMonth;
inline constexpr CardId kInvalidCard = 0xff;
inline constexpr CardMask kFullDeckMask = (CardMask{1} << kCardCount) - 1;

CUGO_HOST_DEVICE constexpr bool is_valid_card(CardId card) noexcept {
  return card < kCardCount;
}

CUGO_HOST_DEVICE constexpr std::uint8_t card_month(CardId card) noexcept {
  return static_cast<std::uint8_t>(card >> 2);
}

CUGO_HOST_DEVICE constexpr std::uint8_t card_slot(CardId card) noexcept {
  return static_cast<std::uint8_t>(card & 0x3u);
}

CUGO_HOST_DEVICE constexpr CardId make_card(std::uint8_t month, std::uint8_t slot) noexcept {
  return static_cast<CardId>((month << 2) | slot);
}

CUGO_HOST_DEVICE constexpr CardMask card_bit(CardId card) noexcept {
  return CardMask{1} << card;
}

CUGO_HOST_DEVICE constexpr CardMask month_mask(std::uint8_t month) noexcept {
  return CardMask{0xf} << (static_cast<unsigned>(month) * kCardsPerMonth);
}

CUGO_HOST_DEVICE constexpr CardMask matching_month_cards(CardMask cards, CardId card) noexcept {
  return cards & month_mask(card_month(card));
}

CUGO_HOST_DEVICE inline int card_count(CardMask mask) noexcept {
#if defined(__CUDA_ARCH__)
  return __popcll(mask);
#else
  return std::popcount(mask);
#endif
}

CUGO_HOST_DEVICE inline CardId first_card(CardMask mask) noexcept {
  if (mask == 0) {
    return kInvalidCard;
  }
#if defined(__CUDA_ARCH__)
  return static_cast<CardId>(__ffsll(static_cast<long long>(mask)) - 1);
#else
  return static_cast<CardId>(std::countr_zero(mask));
#endif
}

CUGO_HOST_DEVICE inline CardId pop_first_card(CardMask& mask) noexcept {
  const CardId card = first_card(mask);
  if (card != kInvalidCard) {
    mask &= mask - 1;
  }
  return card;
}

CUGO_HOST_DEVICE inline CardId select_card_by_rank(CardMask mask, std::uint32_t rank) noexcept {
  if (rank >= static_cast<std::uint32_t>(card_count(mask))) {
    return kInvalidCard;
  }

#if defined(__CUDA_ARCH__)
  const unsigned low = static_cast<unsigned>(mask);
  const unsigned low_count = __popc(low);
  if (rank < low_count) {
    return static_cast<CardId>(__fns(low, 0u, static_cast<int>(rank) + 1));
  }

  const unsigned high = static_cast<unsigned>(mask >> 32);
  const unsigned high_rank = rank - low_count;
  const unsigned bit = __fns(high, 0u, static_cast<int>(high_rank) + 1);
  return bit == 0xffffffffu ? kInvalidCard : static_cast<CardId>(bit + 32u);
#else
  while (rank-- != 0u) {
    mask &= mask - 1;
  }
  return first_card(mask);
#endif
}

}  // namespace cugo::core

#undef CUGO_HOST_DEVICE
