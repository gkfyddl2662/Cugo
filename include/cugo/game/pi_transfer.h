#pragma once

#include <cstdint>

#include "cugo/core/card.h"
#include "cugo/game/gukjin_state.h"

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::game {

using core::CardId;
using core::CardMask;

enum class PiTransferStatus : std::uint8_t {
  kOk = 0,
  kSelectionRequired = 1,
  kInvalidSelection = 2,
};

struct PiTransferSelection { CardMask cards; };
inline constexpr PiTransferSelection kNoPiTransferSelection{0};

struct PiTransferResult {
  PiTransferStatus status;
  std::uint8_t requested_cards;
  std::uint8_t available_cards;
  std::uint8_t transferred_cards;
  std::uint8_t transferred_pi_units;
  CardMask transferred_mask;
};
struct Resolve50WithPiResult {
  Resolve50Result resolve;
  PiTransferResult pi_transfer;
};
struct BonusPlay50WithPiResult {
  Turn50BonusPlayResult play;
  PiTransferResult pi_transfer;
};

CUGO_HOST_DEVICE inline PiTransferResult no_pi_transfer_result() noexcept {
  return PiTransferResult{PiTransferStatus::kOk, 0, 0, 0, 0, 0};
}
CUGO_HOST_DEVICE inline std::uint8_t resolve_pi_steal_card_count(
    const Resolve50Result& result) noexcept {
  if (result.status != Resolve50Status::kOk) return 0;
  unsigned count = static_cast<unsigned>(result.captured_opponent_ppuk) +
                   2u * static_cast<unsigned>(result.captured_own_ppuk);
  if ((result.events & kResolve50EventJjok) != 0) ++count;
  if ((result.events & kResolve50EventTtadak) != 0) ++count;
  if ((result.events & kResolve50EventSweep) != 0) ++count;
  return static_cast<std::uint8_t>(count);
}

CUGO_HOST_DEVICE inline void move_gukjin_role_with_card50(
    TurnState50& state,
    std::uint8_t stealing_actor,
    CardMask transfer,
    bool victim_gukjin_as_double_pi) noexcept {
  if ((transfer & core::card_bit(kGukjin)) == 0) return;
  const std::uint16_t thief_bit = persistent_gukjin_flag50(stealing_actor);
  const std::uint16_t victim_bit =
      persistent_gukjin_flag50(static_cast<std::uint8_t>(stealing_actor ^ 1u));
  state.ppuk_owner1_months &=
      static_cast<std::uint16_t>(~static_cast<std::uint16_t>(thief_bit | victim_bit));
  if (victim_gukjin_as_double_pi) state.ppuk_owner1_months |= thief_bit;
}

CUGO_HOST_DEVICE inline PiTransferResult apply_pi_steal50_impl(
    TurnState50& state,
    std::uint8_t stealing_actor,
    std::uint8_t requested_cards,
    PiTransferSelection selection,
    ScoreOptions victim_options) noexcept {
  stealing_actor &= 1u;
  CardMask& thief = stealing_actor == 0 ? state.captured0 : state.captured1;
  CardMask& victim = stealing_actor == 0 ? state.captured1 : state.captured0;
  const CardMask candidates = pi_card_mask(victim, victim_options);
  const std::uint8_t available =
      static_cast<std::uint8_t>(core::card_count(candidates));

  if (requested_cards == 0 || available == 0) {
    return PiTransferResult{PiTransferStatus::kOk,
                            requested_cards,
                            available,
                            0,
                            0,
                            0};
  }

  const std::uint8_t required =
      requested_cards < available ? requested_cards : available;
  CardMask transfer = 0;
  if (available <= requested_cards) {
    if (selection.cards != 0 && selection.cards != candidates) {
      return PiTransferResult{PiTransferStatus::kInvalidSelection,
                              requested_cards,
                              available,
                              0,
                              0,
                              0};
    }
    transfer = candidates;
  } else {
    if (selection.cards == 0) {
      return PiTransferResult{PiTransferStatus::kSelectionRequired,
                              requested_cards,
                              available,
                              0,
                              0,
                              0};
    }
    if ((selection.cards & ~candidates) != 0 ||
        core::card_count(selection.cards) != required) {
      return PiTransferResult{PiTransferStatus::kInvalidSelection,
                              requested_cards,
                              available,
                              0,
                              0,
                              0};
    }
    transfer = selection.cards;
  }

  const bool moving_gukjin_as_double_pi =
      victim_options.gukjin_as_double_pi &&
      (transfer & core::card_bit(kGukjin)) != 0;
  victim &= ~transfer;
  thief |= transfer;
  move_gukjin_role_with_card50(
      state, stealing_actor, transfer, moving_gukjin_as_double_pi);

  return PiTransferResult{
      PiTransferStatus::kOk,
      requested_cards,
      available,
      static_cast<std::uint8_t>(core::card_count(transfer)),
      pi_units(transfer, victim_options),
      transfer};
}

CUGO_HOST_DEVICE inline PiTransferResult apply_pi_steal50(
    TurnState50& state,
    std::uint8_t stealing_actor,
    std::uint8_t requested_cards,
    PiTransferSelection selection = kNoPiTransferSelection) noexcept {
  const std::uint8_t victim_player =
      static_cast<std::uint8_t>((stealing_actor & 1u) ^ 1u);
  return apply_pi_steal50_impl(
      state, stealing_actor, requested_cards, selection,
      persistent_score_options_for_player50(state, victim_player));
}

CUGO_HOST_DEVICE inline PiTransferResult apply_pi_steal50(
    TurnState50& state,
    std::uint8_t stealing_actor,
    std::uint8_t requested_cards,
    PiTransferSelection selection,
    ScoreOptions opponent_score_options) noexcept {
  return apply_pi_steal50_impl(state, stealing_actor, requested_cards, selection,
                               opponent_score_options);
}

CUGO_HOST_DEVICE inline Resolve50WithPiResult resolve_turn50_with_pi_transfer_impl(
    TurnState50& state,
    Resolve50Choices resolve_choices,
    PiTransferSelection pi_selection,
    bool use_state_options,
    ScoreOptions explicit_options) noexcept {
  TurnState50 next = state;
  const std::uint8_t stealing_actor = state.actor;
  const Resolve50Result resolved = resolve_turn50(next, resolve_choices);
  if (resolved.status != Resolve50Status::kOk)
    return Resolve50WithPiResult{resolved, no_pi_transfer_result()};
  const PiTransferResult transfer = use_state_options
      ? apply_pi_steal50(next, stealing_actor,
                         resolve_pi_steal_card_count(resolved), pi_selection)
      : apply_pi_steal50(next, stealing_actor,
                         resolve_pi_steal_card_count(resolved), pi_selection,
                         explicit_options);
  if (transfer.status != PiTransferStatus::kOk)
    return Resolve50WithPiResult{resolved, transfer};
  state = next;
  return Resolve50WithPiResult{resolved, transfer};
}

CUGO_HOST_DEVICE inline Resolve50WithPiResult resolve_turn50_with_pi_transfer(
    TurnState50& state,
    Resolve50Choices resolve_choices = kNoResolve50Choices,
    PiTransferSelection pi_selection = kNoPiTransferSelection) noexcept {
  return resolve_turn50_with_pi_transfer_impl(
      state, resolve_choices, pi_selection, true, ScoreOptions{});
}

CUGO_HOST_DEVICE inline Resolve50WithPiResult resolve_turn50_with_pi_transfer(
    TurnState50& state,
    Resolve50Choices resolve_choices,
    PiTransferSelection pi_selection,
    ScoreOptions opponent_score_options) noexcept {
  return resolve_turn50_with_pi_transfer_impl(
      state, resolve_choices, pi_selection, false, opponent_score_options);
}

CUGO_HOST_DEVICE inline BonusPlay50WithPiResult
play_bonus_for_turn50_with_pi_transfer_impl(
    TurnState50& state,
    CardId bonus,
    PiTransferSelection pi_selection,
    bool use_state_options,
    ScoreOptions explicit_options) noexcept {
  TurnState50 next = state;
  const std::uint8_t stealing_actor = state.actor;
  const Turn50BonusPlayResult played = play_bonus_for_turn50(next, bonus);
  if (played.status != Turn50Status::kOk)
    return BonusPlay50WithPiResult{played, no_pi_transfer_result()};
  const PiTransferResult transfer = use_state_options
      ? apply_pi_steal50(next, stealing_actor, played.pi_steal_count, pi_selection)
      : apply_pi_steal50(next, stealing_actor, played.pi_steal_count, pi_selection,
                         explicit_options);
  if (transfer.status != PiTransferStatus::kOk)
    return BonusPlay50WithPiResult{played, transfer};
  state = next;
  return BonusPlay50WithPiResult{played, transfer};
}

CUGO_HOST_DEVICE inline BonusPlay50WithPiResult
play_bonus_for_turn50_with_pi_transfer(
    TurnState50& state,
    CardId bonus,
    PiTransferSelection pi_selection = kNoPiTransferSelection) noexcept {
  return play_bonus_for_turn50_with_pi_transfer_impl(
      state, bonus, pi_selection, true, ScoreOptions{});
}

CUGO_HOST_DEVICE inline BonusPlay50WithPiResult
play_bonus_for_turn50_with_pi_transfer(
    TurnState50& state,
    CardId bonus,
    PiTransferSelection pi_selection,
    ScoreOptions opponent_score_options) noexcept {
  return play_bonus_for_turn50_with_pi_transfer_impl(
      state, bonus, pi_selection, false, opponent_score_options);
}

}  // namespace cugo::game

#undef CUGO_HOST_DEVICE
