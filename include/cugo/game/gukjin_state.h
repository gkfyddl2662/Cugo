#pragma once

#include <cstdint>

#include "cugo/core/card.h"
#include "cugo/game/hwatu.h"
#include "cugo/game/turn50.h"

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::game {

using core::CardMask;

inline constexpr std::uint16_t kPersistentGukjinPlayer0Flag50 =
    std::uint16_t{1} << 12;
inline constexpr std::uint16_t kPersistentGukjinPlayer1Flag50 =
    std::uint16_t{1} << 13;
inline constexpr std::uint16_t kPersistentGukjinFlags50 =
    kPersistentGukjinPlayer0Flag50 | kPersistentGukjinPlayer1Flag50;
inline constexpr std::uint16_t kPersistentTurn50MetadataMask =
    kAllMonthBits50 | kPersistentGukjinFlags50;

enum class PersistentGukjinRole50 : std::uint8_t {
  kAnimal = 0,
  kDoublePi = 1,
};

enum class PersistentGukjinRole50Status : std::uint8_t {
  kOk = 0,
  kInvalidPlayer = 1,
  kGukjinNotCaptured = 2,
  kInvalidRole = 3,
};

CUGO_HOST_DEVICE inline CardMask captured_for_player50_persistent(
    const TurnState50& state, std::uint8_t player) noexcept {
  return (player & 1u) == 0 ? state.captured0 : state.captured1;
}

CUGO_HOST_DEVICE inline std::uint16_t persistent_gukjin_flag50(
    std::uint8_t player) noexcept {
  return (player & 1u) == 0 ? kPersistentGukjinPlayer0Flag50
                            : kPersistentGukjinPlayer1Flag50;
}

CUGO_HOST_DEVICE inline PersistentGukjinRole50 persistent_gukjin_role50(
    const TurnState50& state, std::uint8_t player) noexcept {
  return (state.ppuk_owner1_months & persistent_gukjin_flag50(player)) != 0
             ? PersistentGukjinRole50::kDoublePi
             : PersistentGukjinRole50::kAnimal;
}

CUGO_HOST_DEVICE inline ScoreOptions persistent_score_options_for_player50(
    const TurnState50& state, std::uint8_t player) noexcept {
  const CardMask captured = captured_for_player50_persistent(state, player);
  const bool owns_gukjin = (captured & core::card_bit(kGukjin)) != 0;
  return ScoreOptions{
      owns_gukjin &&
      persistent_gukjin_role50(state, player) == PersistentGukjinRole50::kDoublePi};
}

CUGO_HOST_DEVICE inline ScoreBreakdown persistent_score_player50(
    const TurnState50& state, std::uint8_t player) noexcept {
  return score_captured(captured_for_player50_persistent(state, player),
                        persistent_score_options_for_player50(state, player));
}

CUGO_HOST_DEVICE inline PersistentGukjinRole50Status
set_persistent_gukjin_role50(TurnState50& state,
                             std::uint8_t player,
                             PersistentGukjinRole50 role) noexcept {
  if (player > 1u) return PersistentGukjinRole50Status::kInvalidPlayer;
  if (static_cast<std::uint8_t>(role) >
      static_cast<std::uint8_t>(PersistentGukjinRole50::kDoublePi)) {
    return PersistentGukjinRole50Status::kInvalidRole;
  }
  const CardMask captured = captured_for_player50_persistent(state, player);
  if ((captured & core::card_bit(kGukjin)) == 0) {
    return PersistentGukjinRole50Status::kGukjinNotCaptured;
  }
  const std::uint16_t flag = persistent_gukjin_flag50(player);
  if (role == PersistentGukjinRole50::kDoublePi) {
    state.ppuk_owner1_months |= flag;
  } else {
    state.ppuk_owner1_months &= static_cast<std::uint16_t>(~flag);
  }
  return PersistentGukjinRole50Status::kOk;
}

CUGO_HOST_DEVICE inline bool is_valid_persistent_turn_state50(
    const TurnState50& state) noexcept {
  if ((state.ppuk_owner1_months & ~kPersistentTurn50MetadataMask) != 0) {
    return false;
  }
  const std::uint16_t month_owner_bits =
      static_cast<std::uint16_t>(state.ppuk_owner1_months & kAllMonthBits50);
  if ((month_owner_bits & ~state.ppuk_months) != 0) return false;

  const CardMask gukjin = core::card_bit(kGukjin);
  if ((state.ppuk_owner1_months & kPersistentGukjinPlayer0Flag50) != 0 &&
      (state.captured0 & gukjin) == 0) {
    return false;
  }
  if ((state.ppuk_owner1_months & kPersistentGukjinPlayer1Flag50) != 0 &&
      (state.captured1 & gukjin) == 0) {
    return false;
  }

  TurnState50 base = state;
  base.ppuk_owner1_months = month_owner_bits;
  return is_valid_turn_state50(base);
}

}  // namespace cugo::game

#undef CUGO_HOST_DEVICE
