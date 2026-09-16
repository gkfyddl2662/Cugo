#pragma once

#include <cstdint>

#include "cugo/game/special50.h"

#if defined(__CUDACC__)
#define CUGO_HD __host__ __device__
#else
#define CUGO_HD
#endif

namespace cugo::game {

inline constexpr std::uint8_t kNoTerminal50Player = 0xffu;
inline constexpr std::uint64_t kChongtongPoints50 = 10u;
inline constexpr std::uint64_t kThreePpukPoints50 = 7u;
inline constexpr std::uint64_t kThreeConsecutivePpukPoints50 = 49u;

enum class TerminalReason50 : std::uint8_t {
  kNone = 0,
  kStop = 1,
  kInitialChongtong = 2,
  kFloorChongtong = 3,
  kBonusChongtong = 4,
  kThreePpuk = 5,
};

enum class ChongtongAction50 : std::uint8_t {
  kWin = 0,
  kContinue = 1,
};

enum class TerminalStatus50 : std::uint8_t {
  kOk = 0,
  kGameFinished = 1,
  kChongtongChoicePending = 2,
  kNoChongtongChoice = 3,
  kInvalidAction = 4,
};

struct InitialChongtong50 {
  std::uint16_t hand0_months;
  std::uint16_t hand1_months;
  std::uint16_t floor_months;
  std::uint8_t winner;
  TerminalReason50 reason;
};

struct TerminalGameState50 {
  SpecialGameState50 special;
  std::uint8_t ppuk_count0;
  std::uint8_t ppuk_count1;
  std::uint8_t ppuk_streak0;
  std::uint8_t ppuk_streak1;
  std::uint8_t pending_chongtong_actor;
  std::uint16_t pending_chongtong_months;
  std::uint8_t terminal_winner;
  TerminalReason50 terminal_reason;
  std::uint64_t terminal_points;
};

struct TerminalResolve50Result {
  GameResolve50Result game_result;
  std::uint8_t terminal_opened;
};

struct TerminalSpecial50Result {
  SpecialResult50 special_result;
  std::uint8_t terminal_opened;
};

struct TerminalBonus50Result {
  BonusPlay50WithPiResult bonus_result;
  std::uint16_t new_chongtong_months;
  std::uint8_t choice_opened;
};

CUGO_HD inline std::uint16_t chongtong_months50(CardMask cards) noexcept {
  cards &= core::kStandardDeckMask;
  std::uint16_t result = 0;
  for (std::uint8_t month = 0; month < core::kMonthCount; ++month) {
    if ((cards & core::month_mask(month)) == core::month_mask(month)) {
      result |= static_cast<std::uint16_t>(std::uint16_t{1} << month);
    }
  }
  return result;
}

CUGO_HD inline InitialChongtong50 detect_initial_chongtong50(
    const InitialDeal50& deal,
    std::uint8_t first_player) noexcept {
  const std::uint16_t hand0 = chongtong_months50(deal.hand0);
  const std::uint16_t hand1 = chongtong_months50(deal.hand1);
  const std::uint16_t floor = chongtong_months50(deal.floor);
  const std::uint8_t first = static_cast<std::uint8_t>(first_player & 1u);

  if (floor != 0) {
    return InitialChongtong50{hand0, hand1, floor, first,
                              TerminalReason50::kFloorChongtong};
  }
  if (hand0 != 0 && hand1 != 0) {
    return InitialChongtong50{hand0, hand1, floor, first,
                              TerminalReason50::kInitialChongtong};
  }
  if (hand0 != 0) {
    return InitialChongtong50{hand0, hand1, floor, 0,
                              TerminalReason50::kInitialChongtong};
  }
  if (hand1 != 0) {
    return InitialChongtong50{hand0, hand1, floor, 1,
                              TerminalReason50::kInitialChongtong};
  }
  return InitialChongtong50{0, 0, 0, kNoTerminal50Player,
                            TerminalReason50::kNone};
}

CUGO_HD inline bool terminal50_is_finished(
    const TerminalGameState50& state) noexcept {
  return state.terminal_reason != TerminalReason50::kNone;
}

CUGO_HD inline bool terminal50_has_chongtong_choice(
    const TerminalGameState50& state) noexcept {
  return state.pending_chongtong_actor <= 1u;
}

CUGO_HD inline void mark_terminal50(TerminalGameState50& state,
                                    std::uint8_t winner,
                                    TerminalReason50 reason,
                                    std::uint64_t points) noexcept {
  state.terminal_winner = static_cast<std::uint8_t>(winner & 1u);
  state.terminal_reason = reason;
  state.terminal_points = points;
  state.pending_chongtong_actor = kNoTerminal50Player;
  state.pending_chongtong_months = 0;
  state.special.game.decision_actor = kNoGame50Player;
}

CUGO_HD inline TerminalGameState50 make_terminal_game_state50(
    InitialDeal50 deal,
    std::uint8_t first_player = 0) noexcept {
  const InitialChongtong50 initial =
      detect_initial_chongtong50(deal, first_player);
  TerminalGameState50 state{make_special_game_state50(deal, first_player),
                            0,
                            0,
                            0,
                            0,
                            kNoTerminal50Player,
                            0,
                            kNoTerminal50Player,
                            TerminalReason50::kNone,
                            0};
  if (initial.reason != TerminalReason50::kNone) {
    mark_terminal50(state, initial.winner, initial.reason, kChongtongPoints50);
  }
  return state;
}

CUGO_HD inline std::uint8_t ppuk_count50(const TerminalGameState50& state,
                                          std::uint8_t player) noexcept {
  return (player & 1u) == 0 ? state.ppuk_count0 : state.ppuk_count1;
}

CUGO_HD inline std::uint8_t ppuk_streak50(const TerminalGameState50& state,
                                           std::uint8_t player) noexcept {
  return (player & 1u) == 0 ? state.ppuk_streak0 : state.ppuk_streak1;
}

CUGO_HD inline bool record_completed_turn_ppuk50(TerminalGameState50& state,
                                                  std::uint8_t actor,
                                                  bool made_ppuk) noexcept {
  if (terminal50_is_finished(state)) return false;
  actor &= 1u;
  std::uint8_t& count = actor == 0 ? state.ppuk_count0 : state.ppuk_count1;
  std::uint8_t& streak = actor == 0 ? state.ppuk_streak0 : state.ppuk_streak1;
  if (!made_ppuk) {
    streak = 0;
    return false;
  }

  if (count != 0xffu) ++count;
  if (streak != 0xffu) ++streak;
  if (count < 3u) return false;

  const std::uint64_t points = streak >= 3u
      ? kThreeConsecutivePpukPoints50
      : kThreePpukPoints50;
  mark_terminal50(state, actor, TerminalReason50::kThreePpuk, points);
  return true;
}

CUGO_HD inline bool open_bonus_chongtong_choice50(
    TerminalGameState50& state,
    std::uint8_t actor,
    std::uint16_t newly_formed_months) noexcept {
  if (terminal50_is_finished(state) || terminal50_has_chongtong_choice(state) ||
      actor > 1u || newly_formed_months == 0) {
    return false;
  }
  state.pending_chongtong_actor = actor;
  state.pending_chongtong_months =
      static_cast<std::uint16_t>(newly_formed_months & static_cast<std::uint16_t>((std::uint16_t{1} << core::kMonthCount) - 1u));
  return state.pending_chongtong_months != 0;
}

CUGO_HD inline TerminalStatus50 apply_chongtong_action50(
    TerminalGameState50& state,
    ChongtongAction50 action) noexcept {
  if (terminal50_is_finished(state)) return TerminalStatus50::kGameFinished;
  if (!terminal50_has_chongtong_choice(state))
    return TerminalStatus50::kNoChongtongChoice;
  if (static_cast<std::uint8_t>(action) >
      static_cast<std::uint8_t>(ChongtongAction50::kContinue)) {
    return TerminalStatus50::kInvalidAction;
  }

  const std::uint8_t actor = state.pending_chongtong_actor;
  if (action == ChongtongAction50::kWin) {
    mark_terminal50(state, actor, TerminalReason50::kBonusChongtong,
                    kChongtongPoints50);
    return TerminalStatus50::kOk;
  }
  state.pending_chongtong_actor = kNoTerminal50Player;
  state.pending_chongtong_months = 0;
  return TerminalStatus50::kOk;
}

CUGO_HD inline Turn50Status begin_regular_play_terminal50(
    TerminalGameState50& state,
    CardId card) noexcept {
  if (terminal50_is_finished(state) || terminal50_has_chongtong_choice(state))
    return Turn50Status::kWrongPhase;
  return begin_regular_play_game50(state.special.game, card);
}

CUGO_HD inline Turn50Status draw_for_terminal50(
    TerminalGameState50& state) noexcept {
  if (terminal50_is_finished(state) || terminal50_has_chongtong_choice(state))
    return Turn50Status::kWrongPhase;
  return draw_for_game50(state.special.game);
}

CUGO_HD inline TerminalBonus50Result play_bonus_for_terminal50_with_pi_transfer(
    TerminalGameState50& state,
    CardId bonus,
    PiTransferSelection pi_selection = kNoPiTransferSelection) noexcept {
  if (terminal50_is_finished(state) || terminal50_has_chongtong_choice(state)) {
    return TerminalBonus50Result{
        BonusPlay50WithPiResult{
            Turn50BonusPlayResult{Turn50Status::kWrongPhase,
                                  core::kInvalidCard,
                                  0},
            no_pi_transfer_result()},
        0,
        0};
  }

  TerminalGameState50 next = state;
  const std::uint8_t actor = next.special.game.turn.actor;
  const CardMask before_hand = active_hand50(next.special.game.turn);
  const std::uint16_t before = chongtong_months50(before_hand);
  const BonusPlay50WithPiResult result = play_bonus_for_game50_with_pi_transfer(
      next.special.game, bonus, pi_selection);
  if (result.play.status != Turn50Status::kOk ||
      result.pi_transfer.status != PiTransferStatus::kOk) {
    return TerminalBonus50Result{result, 0, 0};
  }

  const CardMask after_hand = active_hand50(next.special.game.turn);
  const std::uint16_t after = chongtong_months50(after_hand);
  const std::uint16_t newly_formed =
      static_cast<std::uint16_t>(after & static_cast<std::uint16_t>(~before));
  const bool opened =
      open_bonus_chongtong_choice50(next, actor, newly_formed);
  state = next;
  return TerminalBonus50Result{result,
                               newly_formed,
                               static_cast<std::uint8_t>(opened ? 1u : 0u)};
}

CUGO_HD inline TerminalResolve50Result resolve_terminal_turn50_with_pi_transfer(
    TerminalGameState50& state,
    Resolve50Choices resolve_choices = kNoResolve50Choices,
    PiTransferSelection pi_selection = kNoPiTransferSelection) noexcept {
  if (terminal50_is_finished(state) || terminal50_has_chongtong_choice(state)) {
    return TerminalResolve50Result{
        GameResolve50Result{
            Resolve50WithPiResult{
                Resolve50Result{Resolve50Status::kWrongPhase,
                                kResolve50EventNone,
                                0,
                                0,
                                0},
                no_pi_transfer_result()},
            kNoGame50Player,
            0,
            0},
        0};
  }

  TerminalGameState50 next = state;
  const std::uint8_t actor = next.special.game.turn.actor;
  const GameResolve50Result result = resolve_game_turn50_with_pi_transfer(
      next.special.game, resolve_choices, pi_selection);
  if (result.turn_result.resolve.status != Resolve50Status::kOk ||
      result.turn_result.pi_transfer.status != PiTransferStatus::kOk) {
    return TerminalResolve50Result{result, 0};
  }

  const bool made_ppuk =
      (result.turn_result.resolve.events & kResolve50EventPpuk) != 0;
  const bool terminal = record_completed_turn_ppuk50(next, actor, made_ppuk);
  GameResolve50Result out = result;
  if (terminal) out.decision_opened = 0;
  state = next;
  return TerminalResolve50Result{out,
                                 static_cast<std::uint8_t>(terminal ? 1u : 0u)};
}

CUGO_HD inline TerminalSpecial50Result apply_terminal_special_action50(
    TerminalGameState50& state,
    SpecialAction50 action) noexcept {
  if (terminal50_is_finished(state) || terminal50_has_chongtong_choice(state)) {
    return TerminalSpecial50Result{
        special_error50(SpecialStatus50::kWrongPhase), 0};
  }

  TerminalGameState50 next = state;
  const std::uint8_t actor = next.special.game.turn.actor;
  const SpecialResult50 result = apply_special_action50(next.special, action);
  if (result.status != SpecialStatus50::kOk) {
    return TerminalSpecial50Result{result, 0};
  }

  if (action.kind != SpecialKind50::kShake) {
    record_completed_turn_ppuk50(next, actor, false);
  }
  state = next;
  return TerminalSpecial50Result{result, 0};
}

CUGO_HD inline GoStopStatus50 apply_terminal_go_stop_decision50(
    TerminalGameState50& state,
    GoStopAction50 action) noexcept {
  if (terminal50_is_finished(state) || terminal50_has_chongtong_choice(state)) {
    return GoStopStatus50::kGameFinished;
  }

  TerminalGameState50 next = state;
  const std::uint8_t actor = next.special.game.decision_actor;
  const GoStopStatus50 status = apply_go_stop_decision50(next.special.game, action);
  if (status != GoStopStatus50::kOk) return status;
  if (action == GoStopAction50::kStop) {
    next.terminal_winner = actor;
    next.terminal_reason = TerminalReason50::kStop;
    next.terminal_points = go_bomb_shake_score50(next.special, actor);
  }
  state = next;
  return status;
}

CUGO_HD inline bool is_valid_terminal_game_state50(
    const TerminalGameState50& state) noexcept {
  if (!is_valid_special_game_state50(state.special)) return false;
  if (state.ppuk_count0 > 3u || state.ppuk_count1 > 3u) return false;
  if (state.ppuk_streak0 > state.ppuk_count0 ||
      state.ppuk_streak1 > state.ppuk_count1) return false;
  if (state.pending_chongtong_actor != kNoTerminal50Player &&
      state.pending_chongtong_actor > 1u) return false;
  if (state.terminal_winner != kNoTerminal50Player &&
      state.terminal_winner > 1u) return false;

  if (terminal50_is_finished(state)) {
    if (state.terminal_winner > 1u || terminal50_has_chongtong_choice(state))
      return false;
    if (state.terminal_reason == TerminalReason50::kStop) {
      if (!game50_is_finished(state.special.game) ||
          state.special.game.winner != state.terminal_winner) return false;
    } else if (game50_is_finished(state.special.game)) {
      return false;
    }
    if (state.terminal_reason == TerminalReason50::kThreePpuk) {
      if (ppuk_count50(state, state.terminal_winner) != 3u) return false;
      const std::uint64_t expected =
          ppuk_streak50(state, state.terminal_winner) == 3u
              ? kThreeConsecutivePpukPoints50
              : kThreePpukPoints50;
      if (state.terminal_points != expected) return false;
    }
    if ((state.terminal_reason == TerminalReason50::kInitialChongtong ||
         state.terminal_reason == TerminalReason50::kFloorChongtong ||
         state.terminal_reason == TerminalReason50::kBonusChongtong) &&
        state.terminal_points != kChongtongPoints50) return false;
  } else {
    if (state.terminal_winner != kNoTerminal50Player ||
        state.terminal_points != 0) return false;
  }

  if (terminal50_has_chongtong_choice(state)) {
    if (state.pending_chongtong_months == 0 ||
        game50_is_finished(state.special.game) ||
        game50_has_pending_decision(state.special.game)) return false;
    const std::uint8_t actor = state.pending_chongtong_actor;
    const CardMask hand = actor == 0 ? state.special.game.turn.hand0
                                     : state.special.game.turn.hand1;
    if ((chongtong_months50(hand) & state.pending_chongtong_months) !=
        state.pending_chongtong_months) return false;
  } else if (state.pending_chongtong_months != 0) {
    return false;
  }

  return true;
}

}  // namespace cugo::game

#undef CUGO_HD
