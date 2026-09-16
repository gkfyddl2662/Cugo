#pragma once

#include <cstdint>

#include "cugo/game/pi_transfer.h"

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::game {

inline constexpr std::uint8_t kNoGame50Player = 0xffu;
inline constexpr std::uint8_t kShinMatgoDecisionBasePoints = 7u;

enum class GoStopAction50 : std::uint8_t {
  kGo = 0,
  kStop = 1,
};

enum class GoStopStatus50 : std::uint8_t {
  kOk = 0,
  kNoDecisionPending = 1,
  kGameFinished = 2,
  kInvalidAction = 3,
};

struct GoAdjustedScore50 {
  std::uint8_t base_points;
  std::uint8_t go_count;
  std::uint8_t additive_go_points;
  std::uint64_t go_multiplier;
  std::uint64_t points_after_go;
};

struct GameState50 {
  TurnState50 turn;
  std::uint8_t go_count0;
  std::uint8_t go_count1;
  std::uint8_t last_go_base_score0;
  std::uint8_t last_go_base_score1;
  std::uint8_t decision_actor;
  std::uint8_t winner;
};

struct GameResolve50Result {
  Resolve50WithPiResult turn_result;
  std::uint8_t completed_actor;
  std::uint8_t base_score;
  std::uint8_t decision_opened;
};

CUGO_HOST_DEVICE inline GameState50 make_game_state50(
    InitialDeal50 deal,
    std::uint8_t first_player = 0) noexcept {
  return GameState50{make_turn_state50(deal, first_player),
                     0,
                     0,
                     0,
                     0,
                     kNoGame50Player,
                     kNoGame50Player};
}

CUGO_HOST_DEVICE inline bool game50_has_pending_decision(
    const GameState50& state) noexcept {
  return state.decision_actor <= 1u;
}

CUGO_HOST_DEVICE inline bool game50_is_finished(
    const GameState50& state) noexcept {
  return state.winner <= 1u;
}

CUGO_HOST_DEVICE inline std::uint8_t go_count_for_player50(
    const GameState50& state,
    std::uint8_t player) noexcept {
  return (player & 1u) == 0 ? state.go_count0 : state.go_count1;
}

CUGO_HOST_DEVICE inline std::uint8_t last_go_base_score_for_player50(
    const GameState50& state,
    std::uint8_t player) noexcept {
  return (player & 1u) == 0 ? state.last_go_base_score0
                            : state.last_go_base_score1;
}

CUGO_HOST_DEVICE inline std::uint8_t game50_base_score(
    const GameState50& state,
    std::uint8_t player) noexcept {
  return persistent_score_player50(state.turn, player).total_points;
}

CUGO_HOST_DEVICE inline std::uint8_t next_go_stop_threshold50(
    const GameState50& state,
    std::uint8_t player) noexcept {
  const std::uint8_t go_count = go_count_for_player50(state, player);
  if (go_count == 0) return kShinMatgoDecisionBasePoints;
  const std::uint8_t last = last_go_base_score_for_player50(state, player);
  return last == 0xffu ? 0xffu : static_cast<std::uint8_t>(last + 1u);
}

CUGO_HOST_DEVICE inline std::uint64_t go_multiplier50(
    std::uint8_t go_count) noexcept {
  std::uint64_t multiplier = 1;
  for (std::uint8_t go = 3; go <= go_count; ++go) {
    multiplier *= 2u;
    if (go == 0xffu) break;
  }
  return multiplier;
}

CUGO_HOST_DEVICE inline GoAdjustedScore50 go_adjusted_score50(
    std::uint8_t base_points,
    std::uint8_t go_count) noexcept {
  const std::uint64_t multiplier = go_multiplier50(go_count);
  const std::uint64_t additive = static_cast<std::uint64_t>(go_count);
  return GoAdjustedScore50{
      base_points,
      go_count,
      go_count,
      multiplier,
      (static_cast<std::uint64_t>(base_points) + additive) * multiplier};
}

CUGO_HOST_DEVICE inline GoAdjustedScore50 go_adjusted_score_player50(
    const GameState50& state,
    std::uint8_t player) noexcept {
  return go_adjusted_score50(game50_base_score(state, player),
                             go_count_for_player50(state, player));
}

CUGO_HOST_DEVICE inline bool maybe_open_go_stop_decision50(
    GameState50& state,
    std::uint8_t completed_actor) noexcept {
  if (completed_actor > 1u || game50_is_finished(state) ||
      game50_has_pending_decision(state)) {
    return false;
  }

  const std::uint8_t base_score = game50_base_score(state, completed_actor);
  const std::uint8_t go_count = go_count_for_player50(state, completed_actor);
  const bool eligible =
      go_count == 0
          ? base_score >= kShinMatgoDecisionBasePoints
          : base_score > last_go_base_score_for_player50(state, completed_actor);
  if (!eligible) return false;

  state.decision_actor = completed_actor;
  return true;
}

CUGO_HOST_DEVICE inline GoStopStatus50 apply_go_stop_decision50(
    GameState50& state,
    GoStopAction50 action) noexcept {
  if (game50_is_finished(state)) return GoStopStatus50::kGameFinished;
  if (!game50_has_pending_decision(state))
    return GoStopStatus50::kNoDecisionPending;
  if (static_cast<std::uint8_t>(action) >
      static_cast<std::uint8_t>(GoStopAction50::kStop)) {
    return GoStopStatus50::kInvalidAction;
  }

  const std::uint8_t actor = state.decision_actor;
  if (action == GoStopAction50::kStop) {
    state.winner = actor;
    state.decision_actor = kNoGame50Player;
    return GoStopStatus50::kOk;
  }

  const std::uint8_t base_score = game50_base_score(state, actor);
  if (actor == 0) {
    ++state.go_count0;
    state.last_go_base_score0 = base_score;
  } else {
    ++state.go_count1;
    state.last_go_base_score1 = base_score;
  }
  state.decision_actor = kNoGame50Player;
  return GoStopStatus50::kOk;
}

CUGO_HOST_DEVICE inline Turn50Status begin_regular_play_game50(
    GameState50& state,
    CardId card) noexcept {
  if (game50_is_finished(state) || game50_has_pending_decision(state))
    return Turn50Status::kWrongPhase;
  return begin_regular_play50(state.turn, card);
}

CUGO_HOST_DEVICE inline Turn50Status draw_for_game50(
    GameState50& state) noexcept {
  if (game50_is_finished(state) || game50_has_pending_decision(state))
    return Turn50Status::kWrongPhase;
  return draw_for_turn50(state.turn);
}

CUGO_HOST_DEVICE inline BonusPlay50WithPiResult play_bonus_for_game50_with_pi_transfer(
    GameState50& state,
    CardId bonus,
    PiTransferSelection pi_selection = kNoPiTransferSelection) noexcept {
  if (game50_is_finished(state) || game50_has_pending_decision(state)) {
    return BonusPlay50WithPiResult{
        Turn50BonusPlayResult{Turn50Status::kWrongPhase, core::kInvalidCard, 0},
        no_pi_transfer_result()};
  }
  return play_bonus_for_turn50_with_pi_transfer(state.turn, bonus, pi_selection);
}

CUGO_HOST_DEVICE inline GameResolve50Result resolve_game_turn50_with_pi_transfer(
    GameState50& state,
    Resolve50Choices resolve_choices = kNoResolve50Choices,
    PiTransferSelection pi_selection = kNoPiTransferSelection) noexcept {
  if (game50_is_finished(state) || game50_has_pending_decision(state)) {
    return GameResolve50Result{
        Resolve50WithPiResult{
            Resolve50Result{Resolve50Status::kWrongPhase,
                            kResolve50EventNone,
                            0,
                            0,
                            0},
            no_pi_transfer_result()},
        kNoGame50Player,
        0,
        0};
  }

  const std::uint8_t completed_actor = state.turn.actor;
  GameState50 next = state;
  const Resolve50WithPiResult result = resolve_turn50_with_pi_transfer(
      next.turn, resolve_choices, pi_selection);
  if (result.resolve.status != Resolve50Status::kOk ||
      result.pi_transfer.status != PiTransferStatus::kOk) {
    return GameResolve50Result{result, completed_actor, 0, 0};
  }

  const std::uint8_t base_score = game50_base_score(next, completed_actor);
  const bool opened = maybe_open_go_stop_decision50(next, completed_actor);
  state = next;
  return GameResolve50Result{result,
                             completed_actor,
                             base_score,
                             static_cast<std::uint8_t>(opened ? 1u : 0u)};
}

CUGO_HOST_DEVICE inline bool is_valid_game_state50(
    const GameState50& state) noexcept {
  if (!is_valid_persistent_turn_state50(state.turn)) return false;
  if (state.decision_actor != kNoGame50Player && state.decision_actor > 1u)
    return false;
  if (state.winner != kNoGame50Player && state.winner > 1u) return false;
  if (game50_is_finished(state) && game50_has_pending_decision(state)) return false;

  if (state.go_count0 == 0 && state.last_go_base_score0 != 0) return false;
  if (state.go_count1 == 0 && state.last_go_base_score1 != 0) return false;
  if (state.go_count0 != 0 &&
      state.last_go_base_score0 < kShinMatgoDecisionBasePoints) {
    return false;
  }
  if (state.go_count1 != 0 &&
      state.last_go_base_score1 < kShinMatgoDecisionBasePoints) {
    return false;
  }

  if (game50_has_pending_decision(state)) {
    if (state.turn.phase != Turn50Phase::kPlay) return false;
    if (state.decision_actor == state.turn.actor) return false;
    const std::uint8_t actor = state.decision_actor;
    const std::uint8_t base_score = game50_base_score(state, actor);
    if (go_count_for_player50(state, actor) == 0) {
      if (base_score < kShinMatgoDecisionBasePoints) return false;
    } else if (base_score <= last_go_base_score_for_player50(state, actor)) {
      return false;
    }
  }

  if (game50_is_finished(state)) {
    if (state.turn.phase != Turn50Phase::kPlay) return false;
    if (state.winner == state.turn.actor) return false;
    if (game50_base_score(state, state.winner) < kShinMatgoDecisionBasePoints)
      return false;
  }

  return true;
}

}  // namespace cugo::game

#undef CUGO_HOST_DEVICE
