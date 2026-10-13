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
// 7 + 14 + 21 immediate first/second/third-ppuk payments plus the 7-point result.
inline constexpr std::uint64_t kThreeConsecutivePpukPoints50 = 49u;
inline constexpr std::int32_t kEventPayment50 = 7;

// SingleServer PNFJMMBGKKM ppuk state per player. Only the first player's turns
// 1, 3 and 5 can chain first/second/third ppuk; any third ppuk ends the game.
enum PpukState50 : std::uint8_t {
  kPpukNone50 = 0,
  kPpukFirst50 = 1,
  kPpukSecond50 = 2,
  kPpukThird50 = 3,
  kPpukOne50 = 4,
  kPpukTwo50 = 5,
  kPpukThree50 = 6,
};

// Hangame Shin Matgo mission types (SingleServer OFFMMMKBBAE).
enum MissionType50 : std::uint8_t {
  kMissionNone50 = 0,
  kMissionEasy50 = 1,      // two cards
  kMissionNormal50 = 2,    // all four cards of a month
  kMissionHard50 = 3,      // a three-card set
  kMissionNoCondition50 = 4,  // multiplier applies to both players outright
  kMissionTriple50 = 6,    // 2^k for k mission cards owned
};

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
  // PpukState50 of each player.
  std::uint8_t ppuk_streak0;
  std::uint8_t ppuk_streak1;
  std::uint8_t pending_chongtong_actor;
  std::uint16_t pending_chongtong_months;
  std::uint8_t terminal_winner;
  TerminalReason50 terminal_reason;
  std::uint64_t terminal_points;
  // Net immediate payments to player 0 (first/second/third ppuk, first ttadak),
  // already scaled by the round multiplier. They stand even on nagari.
  std::int32_t side_points0;
  // Nagari carry-over: every nagari doubles the next game's whole result.
  std::uint8_t round_multiplier;
  std::uint8_t mission_type;
  std::uint8_t mission_multiplier;
  std::uint8_t mission_multi0;
  std::uint8_t mission_multi1;
  // Public history kept for observations (apply_action50 maintains both).
  // Every card a player has held in hand, so held & ~hand is what they have played.
  CardMask held0;
  CardMask held1;
  // Cards a player showed by declaring a shake.
  CardMask revealed0;
  CardMask revealed1;
};

CUGO_HD inline std::uint8_t round_multiplier50(const TerminalGameState50& s) noexcept {
  return s.round_multiplier == 0 ? 1u : s.round_multiplier;
}

CUGO_HD inline std::uint8_t mission_multi50(const TerminalGameState50& s,
                                            std::uint8_t player) noexcept {
  return (player & 1u) == 0 ? s.mission_multi0 : s.mission_multi1;
}

CUGO_HD inline void add_side_payment50(TerminalGameState50& s, std::uint8_t receiver,
                                       std::int32_t base_points) noexcept {
  const std::int32_t amount = base_points * static_cast<std::int32_t>(round_multiplier50(s));
  s.side_points0 += (receiver & 1u) == 0 ? amount : -amount;
}

CUGO_HD inline CardMask hg_cards50(int a, int b, int c = 0, int d = 0) noexcept {
  // Hangame card numbers are 1-based: number n is Cugo CardId n - 1.
  CardMask mask = core::card_bit(static_cast<CardId>(a - 1)) |
                  core::card_bit(static_cast<CardId>(b - 1));
  if (c != 0) mask |= core::card_bit(static_cast<CardId>(c - 1));
  if (d != 0) mask |= core::card_bit(static_cast<CardId>(d - 1));
  return mask;
}

// Mission card sets from SingleServer KCDNCNOCKGB (aEasy/aNomal/aHard/aTriple ids).
CUGO_HD inline CardMask mission_cards50(std::uint8_t type, std::uint8_t index) noexcept {
  if (type == kMissionEasy50) {
    if (index == 10) return hg_cards50(9, 29);  // 38 gwang-ttaeng
    return hg_cards50(4 * index + 1, 4 * index + 2);
  }
  if (type == kMissionNormal50) {
    return core::month_mask(index);
  }
  if (type == kMissionHard50 || type == kMissionTriple50) {
    // The hard list has the "ssangpi" set where the triple list does not.
    if (type == kMissionTriple50 && index >= 4) ++index;
    switch (index) {
      case 0: return hg_cards50(22, 34, 38);
      case 1: return hg_cards50(2, 6, 10);
      case 2: return hg_cards50(14, 18, 26);
      case 3: return hg_cards50(5, 13, 30);
      case 4: return hg_cards50(33, 42, 48);
      case 5: return type == kMissionTriple50 ? hg_cards50(5, 25, 46) : hg_cards50(21, 25, 37);
      case 6: return type == kMissionTriple50 ? hg_cards50(21, 25, 37) : hg_cards50(1, 9, 29);
      case 7: return type == kMissionTriple50 ? hg_cards50(1, 9, 29) : hg_cards50(1, 9, 41);
      case 8: return type == kMissionTriple50 ? hg_cards50(1, 9, 41) : hg_cards50(1, 29, 41);
      case 9: return type == kMissionTriple50 ? hg_cards50(1, 29, 41) : hg_cards50(9, 29, 41);
      case 10: return type == kMissionTriple50 ? hg_cards50(9, 29, 41) : hg_cards50(5, 13, 46);
      case 11: return type == kMissionTriple50 ? hg_cards50(5, 13, 46) : hg_cards50(5, 30, 46);
      case 12: return type == kMissionTriple50 ? hg_cards50(5, 30, 46) : hg_cards50(13, 30, 46);
      case 13: return type == kMissionTriple50 ? hg_cards50(13, 30, 46) : hg_cards50(5, 25, 46);
      default: return 0;
    }
  }
  return 0;
}

// SingleServer OKCHEMHJFHJ: type uniform over easy/normal/hard/no-condition/triple,
// multiplier uniform over 2..4, card set uniform within the type. Drawn from a hash
// of the deal so the stock order is unaffected.
CUGO_HD inline void assign_mission50(TerminalGameState50& s, std::uint64_t seed) noexcept {
  core::SplitMix64 rng{seed ^ 0x6d697373696f6e35ull};
  const std::uint32_t kind = core::uniform_bounded(rng, 5u);
  const std::uint8_t multiplier = static_cast<std::uint8_t>(2u + core::uniform_bounded(rng, 3u));
  std::uint8_t type = kMissionEasy50;
  std::uint32_t options = 11u;
  if (kind == 1u) { type = kMissionNormal50; options = 12u; }
  else if (kind == 2u) { type = kMissionHard50; options = 14u; }
  else if (kind == 3u) { type = kMissionNoCondition50; options = 0u; }
  else if (kind == 4u) { type = kMissionTriple50; options = 13u; }
  const std::uint8_t index = options == 0u ? 0u
      : static_cast<std::uint8_t>(core::uniform_bounded(rng, options));
  s.mission_type = type;
  s.mission_multiplier = type == kMissionTriple50 ? 0u : multiplier;
  s.special.game.turn.mission_cards = mission_cards50(type, index);
  if (type == kMissionNoCondition50) {
    s.mission_multi0 = multiplier;
    s.mission_multi1 = multiplier;
  }
}

// SingleServer EKGDFAHDPJD after the actor acquires cards (captures or steals).
CUGO_HD inline void update_missions50(TerminalGameState50& s, std::uint8_t actor,
                                      CardMask acquired) noexcept {
  const TurnState50& t = s.special.game.turn;
  const CardMask mine = (actor & 1u) == 0 ? t.captured0 : t.captured1;
  std::uint8_t& multi = (actor & 1u) == 0 ? s.mission_multi0 : s.mission_multi1;
  if (s.mission_type == kMissionTriple50) {
    multi = static_cast<std::uint8_t>(1u << core::card_count(t.mission_cards & mine));
    return;
  }
  if (t.mission_cards == 0 || (acquired & t.mission_cards) == 0) return;
  if ((t.mission_cards & ~mine) == 0) multi = s.mission_multiplier;
}

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
  state.special.game.gukjin_decision = 0;
}

CUGO_HD inline TerminalGameState50 make_terminal_game_state50(
    InitialDeal50 deal,
    std::uint8_t first_player = 0,
    std::uint8_t round_multiplier = 1,
    bool missions = true) noexcept {
  TerminalGameState50 state{make_special_game_state50(deal, first_player),
                            0,
                            0,
                            0,
                            0,
                            kNoTerminal50Player,
                            0,
                            kNoTerminal50Player,
                            TerminalReason50::kNone,
                            0,
                            0,
                            static_cast<std::uint8_t>(round_multiplier == 0 ? 1 : round_multiplier),
                            kMissionNone50,
                            0,
                            0,
                            0};
  if (missions) assign_mission50(state, deal.rng_state);
  // Chongtong is judged on the dealt hands and the refilled floor.
  const TurnState50& turn = state.special.game.turn;
  InitialDeal50 dealt = deal;
  dealt.floor = turn.floor;
  const InitialChongtong50 initial = detect_initial_chongtong50(dealt, first_player);
  if (initial.reason != TerminalReason50::kNone) {
    mark_terminal50(state, initial.winner, initial.reason, kChongtongPoints50);
  }
  return state;
}

CUGO_HD inline void sync_game_stop50(TerminalGameState50& state) noexcept {
  if (state.terminal_reason != TerminalReason50::kNone ||
      !game50_is_finished(state.special.game)) return;
  const std::uint8_t winner = state.special.game.winner;
  state.terminal_winner = winner;
  state.terminal_reason = TerminalReason50::kStop;
  state.terminal_points = go_bomb_shake_score50(state.special, winner);
}

CUGO_HD inline std::uint8_t ppuk_count50(const TerminalGameState50& state,
                                          std::uint8_t player) noexcept {
  return (player & 1u) == 0 ? state.ppuk_count0 : state.ppuk_count1;
}

CUGO_HD inline std::uint8_t ppuk_streak50(const TerminalGameState50& state,
                                           std::uint8_t player) noexcept {
  return (player & 1u) == 0 ? state.ppuk_streak0 : state.ppuk_streak1;
}

// Call after the actor's turn completed (turn_index already advanced, so it equals
// Hangame's 1-based number of that turn). Pays 7/14/21 for first/second/third ppuk
// and ends the game at the third ppuk.
CUGO_HD inline bool record_completed_turn_ppuk50(TerminalGameState50& state,
                                                  std::uint8_t actor,
                                                  bool made_ppuk) noexcept {
  if (terminal50_is_finished(state) || !made_ppuk) return false;
  actor &= 1u;
  std::uint8_t& count = actor == 0 ? state.ppuk_count0 : state.ppuk_count1;
  std::uint8_t& ppuk = actor == 0 ? state.ppuk_streak0 : state.ppuk_streak1;
  const std::uint16_t turn = state.special.game.turn.turn_index;
  if (count != 0xffu) ++count;
  switch (ppuk) {
    case kPpukNone50: ppuk = turn <= 1u ? kPpukFirst50 : kPpukOne50; break;
    case kPpukFirst50: ppuk = turn <= 3u ? kPpukSecond50 : kPpukTwo50; break;
    case kPpukSecond50: ppuk = turn <= 5u ? kPpukThird50 : kPpukThree50; break;
    case kPpukOne50: ppuk = kPpukTwo50; break;
    case kPpukTwo50: ppuk = kPpukThree50; break;
    default: break;
  }
  if (ppuk == kPpukFirst50 || ppuk == kPpukSecond50 || ppuk == kPpukThird50) {
    add_side_payment50(state, actor, kEventPayment50 * static_cast<std::int32_t>(ppuk));
  }
  if (ppuk != kPpukThird50 && ppuk != kPpukThree50) return false;
  mark_terminal50(state, actor, TerminalReason50::kThreePpuk, kThreePpukPoints50);
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

  if (result.pi_transfer.transferred_mask != 0) {
    update_missions50(next, actor, result.pi_transfer.transferred_mask);
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
  const CardMask before = actor == 0 ? next.special.game.turn.captured0
                                     : next.special.game.turn.captured1;
  const GameResolve50Result result = resolve_game_turn50_with_pi_transfer(
      next.special.game, resolve_choices, pi_selection);
  if (result.turn_result.resolve.status != Resolve50Status::kOk ||
      result.turn_result.pi_transfer.status != PiTransferStatus::kOk) {
    return TerminalResolve50Result{result, 0};
  }

  const std::uint8_t events = result.turn_result.resolve.events;
  // SingleServer DDADAK on turn 1: the first player's opening ttadak pays 7.
  if ((events & kResolve50EventTtadak) != 0 && next.special.game.turn.turn_index == 1u) {
    add_side_payment50(next, actor, kEventPayment50);
  }
  const CardMask after = actor == 0 ? next.special.game.turn.captured0
                                    : next.special.game.turn.captured1;
  update_missions50(next, actor, after & ~before);
  const bool made_ppuk = (events & kResolve50EventPpuk) != 0;
  const bool terminal = record_completed_turn_ppuk50(next, actor, made_ppuk);
  sync_game_stop50(next);
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
  const CardMask before = actor == 0 ? next.special.game.turn.captured0
                                     : next.special.game.turn.captured1;
  const SpecialResult50 result = apply_special_action50(next.special, action);
  if (result.status != SpecialStatus50::kOk) {
    return TerminalSpecial50Result{result, 0};
  }

  if (action.kind != SpecialKind50::kShake) {
    const CardMask after = actor == 0 ? next.special.game.turn.captured0
                                      : next.special.game.turn.captured1;
    update_missions50(next, actor, after & ~before);
    sync_game_stop50(next);
  }
  state = next;
  return TerminalSpecial50Result{result, 0};
}

CUGO_HD inline GoStopStatus50 apply_terminal_gukjin_decision50(
    TerminalGameState50& state,
    bool as_pi) noexcept {
  if (terminal50_is_finished(state) || terminal50_has_chongtong_choice(state)) {
    return GoStopStatus50::kGameFinished;
  }
  TerminalGameState50 next = state;
  const GoStopStatus50 status = apply_gukjin_decision50(next.special.game, as_pi);
  if (status != GoStopStatus50::kOk) return status;
  sync_game_stop50(next);
  state = next;
  return status;
}

CUGO_HD inline GoStopStatus50 apply_terminal_go_stop_decision50(
    TerminalGameState50& state,
    GoStopAction50 action) noexcept {
  if (terminal50_is_finished(state) || terminal50_has_chongtong_choice(state)) {
    return GoStopStatus50::kGameFinished;
  }

  TerminalGameState50 next = state;
  const GoStopStatus50 status = apply_go_stop_decision50(next.special.game, action);
  if (status != GoStopStatus50::kOk) return status;
  sync_game_stop50(next);
  state = next;
  return status;
}

CUGO_HD inline bool is_valid_terminal_game_state50(
    const TerminalGameState50& state) noexcept {
  if (!is_valid_special_game_state50(state.special)) return false;
  if (state.ppuk_count0 > 3u || state.ppuk_count1 > 3u) return false;
  for (std::uint8_t p = 0; p < 2u; ++p) {
    const std::uint8_t ppuk = ppuk_streak50(state, p);
    if (ppuk > kPpukThree50) return false;
    const std::uint8_t expected = ppuk == kPpukNone50 ? 0u
        : ppuk <= kPpukThird50 ? ppuk : static_cast<std::uint8_t>(ppuk - 3u);
    if (ppuk_count50(state, p) != expected) return false;
  }
  if (state.round_multiplier == 0) return false;
  if (state.special.game.turn.mission_cards != 0 &&
      state.mission_type != kMissionEasy50 && state.mission_type != kMissionNormal50 &&
      state.mission_type != kMissionHard50 && state.mission_type != kMissionTriple50)
    return false;
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
      if (ppuk_count50(state, state.terminal_winner) != 3u ||
          state.terminal_points != kThreePpukPoints50) return false;
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
        game50_has_pending_choice(state.special.game)) return false;
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
