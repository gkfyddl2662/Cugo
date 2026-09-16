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

}  // namespace cugo::core

#undef CUGO_HOST_DEVICE
