#pragma once

#include <cstdint>

#include "cugo/core/card.h"

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::game {

using core::CardId;
using core::CardMask;

inline constexpr CardMask kBrightMask =
    core::card_bit(core::make_card(0, 0)) |
    core::card_bit(core::make_card(2, 0)) |
    core::card_bit(core::make_card(7, 0)) |
    core::card_bit(core::make_card(10, 0)) |
    core::card_bit(core::make_card(11, 0));

inline constexpr CardId kRainBright = core::make_card(11, 0);

inline constexpr CardMask kAnimalMask =
    core::card_bit(core::make_card(1, 0)) |
    core::card_bit(core::make_card(3, 0)) |
    core::card_bit(core::make_card(4, 0)) |
    core::card_bit(core::make_card(5, 0)) |
    core::card_bit(core::make_card(6, 0)) |
    core::card_bit(core::make_card(7, 1)) |
    core::card_bit(core::make_card(8, 0)) |
    core::card_bit(core::make_card(9, 0)) |
    core::card_bit(core::make_card(11, 1));

inline constexpr CardId kGukjin = core::make_card(8, 0);

inline constexpr CardMask kRibbonMask =
    core::card_bit(core::make_card(0, 1)) |
    core::card_bit(core::make_card(1, 1)) |
    core::card_bit(core::make_card(2, 1)) |
    core::card_bit(core::make_card(3, 1)) |
    core::card_bit(core::make_card(4, 1)) |
    core::card_bit(core::make_card(5, 1)) |
    core::card_bit(core::make_card(6, 1)) |
    core::card_bit(core::make_card(8, 1)) |
    core::card_bit(core::make_card(9, 1)) |
    core::card_bit(core::make_card(11, 2));

inline constexpr CardMask kPlainPiMask =
    core::card_bit(core::make_card(0, 2)) | core::card_bit(core::make_card(0, 3)) |
    core::card_bit(core::make_card(1, 2)) | core::card_bit(core::make_card(1, 3)) |
    core::card_bit(core::make_card(2, 2)) | core::card_bit(core::make_card(2, 3)) |
    core::card_bit(core::make_card(3, 2)) | core::card_bit(core::make_card(3, 3)) |
    core::card_bit(core::make_card(4, 2)) | core::card_bit(core::make_card(4, 3)) |
    core::card_bit(core::make_card(5, 2)) | core::card_bit(core::make_card(5, 3)) |
    core::card_bit(core::make_card(6, 2)) | core::card_bit(core::make_card(6, 3)) |
    core::card_bit(core::make_card(7, 2)) | core::card_bit(core::make_card(7, 3)) |
    core::card_bit(core::make_card(8, 2)) | core::card_bit(core::make_card(8, 3)) |
    core::card_bit(core::make_card(9, 2)) | core::card_bit(core::make_card(9, 3)) |
    core::card_bit(core::make_card(10, 2)) | core::card_bit(core::make_card(10, 3));

inline constexpr CardMask kFixedDoublePiMask =
    core::card_bit(core::make_card(10, 1)) |
    core::card_bit(core::make_card(11, 3));

inline constexpr CardMask kGodoriMask =
    core::card_bit(core::make_card(1, 0)) |
    core::card_bit(core::make_card(3, 0)) |
    core::card_bit(core::make_card(7, 1));

inline constexpr CardMask kHongdanMask =
    core::card_bit(core::make_card(0, 1)) |
    core::card_bit(core::make_card(1, 1)) |
    core::card_bit(core::make_card(2, 1));

inline constexpr CardMask kChodanMask =
    core::card_bit(core::make_card(3, 1)) |
    core::card_bit(core::make_card(4, 1)) |
    core::card_bit(core::make_card(6, 1));

inline constexpr CardMask kCheongdanMask =
    core::card_bit(core::make_card(5, 1)) |
    core::card_bit(core::make_card(8, 1)) |
    core::card_bit(core::make_card(9, 1));

inline constexpr CardMask kPrimaryClassMask =
    kBrightMask | kAnimalMask | kRibbonMask | kPlainPiMask | kFixedDoublePiMask;

static_assert(kPrimaryClassMask == core::kStandardDeckMask);
static_assert((kBrightMask & kAnimalMask) == 0);
static_assert((kBrightMask & kRibbonMask) == 0);
static_assert((kAnimalMask & kRibbonMask) == 0);
static_assert((kPlainPiMask & kFixedDoublePiMask) == 0);

struct ScoreOptions {
  bool gukjin_as_double_pi = false;
};

enum ScoreFlag : std::uint8_t {
  kScoreFlagNone = 0,
  kScoreFlagGodori = 1u << 0,
  kScoreFlagHongdan = 1u << 1,
  kScoreFlagChodan = 1u << 2,
  kScoreFlagCheongdan = 1u << 3,
  kScoreFlagMeongtta = 1u << 4,
};

struct ScoreBreakdown {
  std::uint8_t bright_count;
  std::uint8_t animal_count;
  std::uint8_t ribbon_count;
  std::uint8_t pi_units;
  std::uint8_t bright_points;
  std::uint8_t animal_points;
  std::uint8_t ribbon_points;
  std::uint8_t pi_points;
  std::uint8_t flags;
  std::uint8_t total_points;
};

CUGO_HOST_DEVICE constexpr bool contains_all(CardMask cards, CardMask subset) noexcept {
  return (cards & subset) == subset;
}

CUGO_HOST_DEVICE inline std::uint8_t count_u8(CardMask cards) noexcept {
  return static_cast<std::uint8_t>(core::card_count(cards));
}

CUGO_HOST_DEVICE inline CardMask pi_card_mask(
    CardMask cards, ScoreOptions options = {}) noexcept {
  CardMask mask = cards & (kPlainPiMask | kFixedDoublePiMask | core::kBonusCardMask);
  if (options.gukjin_as_double_pi) {
    mask |= cards & core::card_bit(kGukjin);
  }
  return mask;
}

CUGO_HOST_DEVICE inline std::uint8_t pi_units(CardMask cards,
                                              ScoreOptions options = {}) noexcept {
  unsigned units = static_cast<unsigned>(core::card_count(cards & kPlainPiMask));
  units += 2u * static_cast<unsigned>(core::card_count(cards & kFixedDoublePiMask));
  if ((cards & core::card_bit(core::kBonusTwoPi)) != 0) units += 2u;
  if ((cards & core::card_bit(core::kBonusThreePi)) != 0) units += 3u;
  if (options.gukjin_as_double_pi && (cards & core::card_bit(kGukjin)) != 0) {
    units += 2u;
  }
  return static_cast<std::uint8_t>(units);
}

CUGO_HOST_DEVICE inline ScoreBreakdown score_captured(
    CardMask cards, ScoreOptions options = {}) noexcept {
  const std::uint8_t bright_count = count_u8(cards & kBrightMask);
  CardMask animals = cards & kAnimalMask;
  if (options.gukjin_as_double_pi) animals &= ~core::card_bit(kGukjin);
  const std::uint8_t animal_count = count_u8(animals);
  const std::uint8_t ribbon_count = count_u8(cards & kRibbonMask);
  const std::uint8_t pi_count = pi_units(cards, options);

  std::uint8_t bright_points = 0;
  if (bright_count == 3) {
    bright_points = (cards & core::card_bit(kRainBright)) != 0 ? 2u : 3u;
  } else if (bright_count == 4) {
    bright_points = 4;
  } else if (bright_count == 5) {
    bright_points = 15;
  }

  std::uint8_t flags = kScoreFlagNone;
  std::uint8_t animal_points = animal_count >= 5 ? animal_count - 4 : 0;
  if (contains_all(animals, kGodoriMask)) {
    flags |= kScoreFlagGodori;
    animal_points = static_cast<std::uint8_t>(animal_points + 5u);
  }
  if (animal_count >= 7) flags |= kScoreFlagMeongtta;

  std::uint8_t ribbon_points = ribbon_count >= 5 ? ribbon_count - 4 : 0;
  if (contains_all(cards, kHongdanMask)) {
    flags |= kScoreFlagHongdan;
    ribbon_points = static_cast<std::uint8_t>(ribbon_points + 3u);
  }
  if (contains_all(cards, kChodanMask)) {
    flags |= kScoreFlagChodan;
    ribbon_points = static_cast<std::uint8_t>(ribbon_points + 3u);
  }
  if (contains_all(cards, kCheongdanMask)) {
    flags |= kScoreFlagCheongdan;
    ribbon_points = static_cast<std::uint8_t>(ribbon_points + 3u);
  }

  const std::uint8_t pi_points = pi_count >= 10 ? pi_count - 9 : 0;
  const unsigned total = static_cast<unsigned>(bright_points) + animal_points +
                         ribbon_points + pi_points;
  return ScoreBreakdown{bright_count, animal_count, ribbon_count, pi_count,
                        bright_points, animal_points, ribbon_points, pi_points,
                        flags, static_cast<std::uint8_t>(total)};
}

}  // namespace cugo::game

#undef CUGO_HOST_DEVICE
