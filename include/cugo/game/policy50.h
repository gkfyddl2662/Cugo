#pragma once

#include <cstdint>

#include "cugo/game/rollout50.h"

#if defined(__CUDACC__)
#define CUGO_HD __host__ __device__
#else
#define CUGO_HD
#endif

namespace cugo::game {

inline constexpr std::uint16_t kPolicyPlayCardBegin50 = 0;
inline constexpr std::uint16_t kPolicyPlayBonusBegin50 = 48;
inline constexpr std::uint16_t kPolicyShakeBegin50 = 50;
inline constexpr std::uint16_t kPolicyBombBegin50 = 98;
inline constexpr std::uint16_t kPolicyGrenadeBegin50 = 110;
inline constexpr std::uint16_t kPolicyBombCredit50 = 122;
inline constexpr std::uint16_t kPolicyGo50 = 123;
inline constexpr std::uint16_t kPolicyStop50 = 124;
inline constexpr std::uint16_t kPolicyChongtongWin50 = 125;
inline constexpr std::uint16_t kPolicyChongtongContinue50 = 126;
inline constexpr std::uint16_t kPolicyActionCount50 = 127;
inline constexpr std::uint16_t kInvalidPolicyAction50 = 0xffffu;

static_assert(kPolicyChongtongContinue50 + 1u == kPolicyActionCount50);

enum class PolicyDecision50 : std::uint8_t {
  kNone = 0,
  kPrimary = 1,
  kGoStop = 2,
  kChongtong = 3,
};

struct PolicyActionMask50 {
  std::uint64_t lo;
  std::uint64_t hi;
};

struct PolicyObservation50 {
  CardMask own_hand;
  CardMask floor;
  CardMask own_captured;
  CardMask opponent_captured;
  CardMask pending_public;
  CardMask unseen;

  std::uint16_t own_shaken_months;
  std::uint16_t opponent_shaken_months;
  std::uint16_t ppuk_months;
  std::uint16_t own_ppuk_months;
  std::uint16_t opponent_ppuk_months;
  std::uint16_t bonus2_ppuk_months;
  std::uint16_t bonus3_ppuk_months;
  std::uint16_t pending_chongtong_months;
  std::uint16_t turn_index;

  std::uint8_t decision_kind;
  std::uint8_t stock_count;
  std::uint8_t own_go_count;
  std::uint8_t opponent_go_count;
  std::uint8_t own_last_go_base_score;
  std::uint8_t opponent_last_go_base_score;
  std::uint8_t own_base_score;
  std::uint8_t opponent_base_score;
  std::uint8_t own_bombs;
  std::uint8_t opponent_bombs;
  std::uint8_t own_credits;
  std::uint8_t opponent_credits;
  std::uint8_t own_ppuk_count;
  std::uint8_t opponent_ppuk_count;
  std::uint8_t own_ppuk_streak;
  std::uint8_t opponent_ppuk_streak;
  std::uint8_t own_gukjin_double_pi;
  std::uint8_t opponent_gukjin_double_pi;
};

struct PolicyPacket50 {
  PolicyObservation50 observation;
  PolicyActionMask50 legal_actions;
};

CUGO_HD inline PolicyActionMask50 empty_policy_action_mask50() noexcept {
  return PolicyActionMask50{0, 0};
}

CUGO_HD inline void set_policy_action50(PolicyActionMask50& mask,
                                         std::uint16_t action) noexcept {
  if (action >= kPolicyActionCount50) return;
  if (action < 64u) mask.lo |= std::uint64_t{1} << action;
  else mask.hi |= std::uint64_t{1} << (action - 64u);
}

CUGO_HD inline bool has_policy_action50(PolicyActionMask50 mask,
                                        std::uint16_t action) noexcept {
  if (action >= kPolicyActionCount50) return false;
  if (action < 64u) return (mask.lo & (std::uint64_t{1} << action)) != 0;
  return (mask.hi & (std::uint64_t{1} << (action - 64u))) != 0;
}

CUGO_HD inline std::uint16_t policy_action_count50(
    PolicyActionMask50 mask) noexcept {
  return static_cast<std::uint16_t>(core::card_count(mask.lo) +
                                    core::card_count(mask.hi));
}

CUGO_HD inline PolicyDecision50 policy_decision_kind50(
    const TerminalGameState50& state) noexcept {
  if (terminal50_is_finished(state)) return PolicyDecision50::kNone;
  if (terminal50_has_chongtong_choice(state)) return PolicyDecision50::kChongtong;
  if (game50_has_pending_decision(state.special.game)) return PolicyDecision50::kGoStop;
  if (state.special.game.turn.phase == Turn50Phase::kPlay)
    return PolicyDecision50::kPrimary;
  return PolicyDecision50::kNone;
}

CUGO_HD inline std::uint8_t policy_decision_player50(
    const TerminalGameState50& state) noexcept {
  if (terminal50_is_finished(state)) return kNoTerminal50Player;
  if (terminal50_has_chongtong_choice(state)) return state.pending_chongtong_actor;
  if (game50_has_pending_decision(state.special.game))
    return state.special.game.decision_actor;
  return state.special.game.turn.actor;
}

CUGO_HD inline PolicyActionMask50 legal_primary_actions50(
    const TerminalGameState50& state) noexcept {
  PolicyActionMask50 mask = empty_policy_action_mask50();
  const PolicyDecision50 decision = policy_decision_kind50(state);

  if (decision == PolicyDecision50::kChongtong) {
    set_policy_action50(mask, kPolicyChongtongWin50);
    set_policy_action50(mask, kPolicyChongtongContinue50);
    return mask;
  }
  if (decision == PolicyDecision50::kGoStop) {
    set_policy_action50(mask, kPolicyGo50);
    set_policy_action50(mask, kPolicyStop50);
    return mask;
  }
  if (decision != PolicyDecision50::kPrimary) return mask;

  const TurnState50& turn = state.special.game.turn;
  const std::uint8_t actor = turn.actor;
  const CardMask hand = active_hand50(turn);
  if (turn.stock == 0) return mask;

  CardMask standard = hand & core::kStandardDeckMask;
  while (standard != 0) {
    const CardId card = core::pop_first_card(standard);
    set_policy_action50(mask, static_cast<std::uint16_t>(card));
  }

  if ((hand & core::card_bit(core::kBonusTwoPi)) != 0)
    set_policy_action50(mask, kPolicyPlayBonusBegin50);
  if ((hand & core::card_bit(core::kBonusThreePi)) != 0)
    set_policy_action50(mask, static_cast<std::uint16_t>(kPolicyPlayBonusBegin50 + 1u));

  const std::uint16_t shake_months = legal_shake_months50(state.special);
  CardMask shake_cards = hand & core::kStandardDeckMask;
  while (shake_cards != 0) {
    const CardId card = core::pop_first_card(shake_cards);
    const std::uint16_t month_bit = static_cast<std::uint16_t>(
        std::uint16_t{1} << core::card_month(card));
    if ((shake_months & month_bit) != 0) {
      set_policy_action50(mask,
                          static_cast<std::uint16_t>(kPolicyShakeBegin50 + card));
    }
  }

  const std::uint16_t bomb_months = legal_bomb_months50(state.special);
  const std::uint16_t grenade_months = legal_grenade_months50(state.special);
  for (std::uint8_t month = 0; month < core::kMonthCount; ++month) {
    const std::uint16_t bit = static_cast<std::uint16_t>(std::uint16_t{1} << month);
    if ((bomb_months & bit) != 0)
      set_policy_action50(mask,
                          static_cast<std::uint16_t>(kPolicyBombBegin50 + month));
    if ((grenade_months & bit) != 0)
      set_policy_action50(mask,
                          static_cast<std::uint16_t>(kPolicyGrenadeBegin50 + month));
  }

  if (credits50(state.special, actor) != 0)
    set_policy_action50(mask, kPolicyBombCredit50);
  return mask;
}

CUGO_HD inline std::uint16_t primary_action_index50(
    const Action50& action) noexcept {
  switch (action.kind) {
    case ActionKind50::kPlayCard:
      return core::is_standard_card(action.card)
          ? static_cast<std::uint16_t>(action.card)
          : kInvalidPolicyAction50;
    case ActionKind50::kPlayBonus:
      if (action.card == core::kBonusTwoPi) return kPolicyPlayBonusBegin50;
      if (action.card == core::kBonusThreePi)
        return static_cast<std::uint16_t>(kPolicyPlayBonusBegin50 + 1u);
      return kInvalidPolicyAction50;
    case ActionKind50::kShake:
      return core::is_standard_card(action.card)
          ? static_cast<std::uint16_t>(kPolicyShakeBegin50 + action.card)
          : kInvalidPolicyAction50;
    case ActionKind50::kBomb:
      return action.month < core::kMonthCount
          ? static_cast<std::uint16_t>(kPolicyBombBegin50 + action.month)
          : kInvalidPolicyAction50;
    case ActionKind50::kGrenade:
      return action.month < core::kMonthCount
          ? static_cast<std::uint16_t>(kPolicyGrenadeBegin50 + action.month)
          : kInvalidPolicyAction50;
    case ActionKind50::kBombCredit: return kPolicyBombCredit50;
    case ActionKind50::kGo: return kPolicyGo50;
    case ActionKind50::kStop: return kPolicyStop50;
    case ActionKind50::kChongtongWin: return kPolicyChongtongWin50;
    case ActionKind50::kChongtongContinue: return kPolicyChongtongContinue50;
    default: return kInvalidPolicyAction50;
  }
}

CUGO_HD inline ActionStatus50 decode_primary_action50(
    const TerminalGameState50& state,
    std::uint16_t index,
    Action50& action) noexcept {
  const PolicyActionMask50 legal = legal_primary_actions50(state);
  if (!has_policy_action50(legal, index)) {
    action = kInvalidAction50;
    return ActionStatus50::kNotLegal;
  }

  if (index < kPolicyPlayBonusBegin50) {
    action = make_action50(ActionKind50::kPlayCard);
    action.card = static_cast<CardId>(index);
    return canonicalize_regular_action50(state, action, false);
  }
  if (index < kPolicyShakeBegin50) {
    action = make_action50(ActionKind50::kPlayBonus);
    action.card = index == kPolicyPlayBonusBegin50
        ? core::kBonusTwoPi : core::kBonusThreePi;
    return canonicalize_bonus_action50(state, action);
  }
  if (index < kPolicyBombBegin50) {
    action = make_action50(ActionKind50::kShake);
    action.card = static_cast<CardId>(index - kPolicyShakeBegin50);
    action.month = core::card_month(action.card);
    return canonicalize_regular_action50(state, action, true);
  }
  if (index < kPolicyGrenadeBegin50) {
    action = make_action50(ActionKind50::kBomb);
    action.month = static_cast<std::uint8_t>(index - kPolicyBombBegin50);
    return canonicalize_special_action50(state, action);
  }
  if (index < kPolicyBombCredit50) {
    action = make_action50(ActionKind50::kGrenade);
    action.month = static_cast<std::uint8_t>(index - kPolicyGrenadeBegin50);
    return canonicalize_special_action50(state, action);
  }
  if (index == kPolicyBombCredit50) {
    action = make_action50(ActionKind50::kBombCredit);
    return canonicalize_special_action50(state, action);
  }
  if (index == kPolicyGo50) {
    action = make_action50(ActionKind50::kGo);
    return ActionStatus50::kOk;
  }
  if (index == kPolicyStop50) {
    action = make_action50(ActionKind50::kStop);
    return ActionStatus50::kOk;
  }
  if (index == kPolicyChongtongWin50) {
    action = make_action50(ActionKind50::kChongtongWin);
    return ActionStatus50::kOk;
  }
  action = make_action50(ActionKind50::kChongtongContinue);
  return ActionStatus50::kOk;
}

CUGO_HD inline CardMask pending_public_cards50(const TurnState50& turn) noexcept {
  CardMask cards = turn.pending_bonus_mask;
  if (core::is_standard_card(turn.pending_played))
    cards |= core::card_bit(turn.pending_played);
  if (core::is_standard_card(turn.pending_drawn))
    cards |= core::card_bit(turn.pending_drawn);
  return cards;
}

CUGO_HD inline PolicyObservation50 pack_policy_observation50(
    const TerminalGameState50& state) noexcept {
  const TurnState50& turn = state.special.game.turn;
  std::uint8_t player = policy_decision_player50(state);
  if (player > 1u) player = static_cast<std::uint8_t>(turn.actor & 1u);
  const std::uint8_t opponent = static_cast<std::uint8_t>(player ^ 1u);

  const CardMask own_hand = player == 0 ? turn.hand0 : turn.hand1;
  const CardMask own_captured = player == 0 ? turn.captured0 : turn.captured1;
  const CardMask opponent_captured = player == 0 ? turn.captured1 : turn.captured0;
  const CardMask pending_public = pending_public_cards50(turn);
  const CardMask public_known = own_hand | turn.floor | own_captured |
                                opponent_captured | pending_public;
  const CardMask unseen = core::kShinMatgoDeckMask & ~public_known;

  const std::uint16_t owner1_months = static_cast<std::uint16_t>(
      turn.ppuk_owner1_months & kAllMonthBits50);
  const std::uint16_t owner0_months = static_cast<std::uint16_t>(
      turn.ppuk_months & static_cast<std::uint16_t>(~owner1_months));
  const std::uint16_t own_ppuk = player == 0 ? owner0_months : owner1_months;
  const std::uint16_t opponent_ppuk = player == 0 ? owner1_months : owner0_months;

  const std::uint16_t own_shaken = player == 0 ? state.special.shaken0
                                                : state.special.shaken1;
  const std::uint16_t opponent_shaken = player == 0 ? state.special.shaken1
                                                     : state.special.shaken0;
  const ScoreBreakdown own_score = persistent_score_player50(turn, player);
  const ScoreBreakdown opponent_score = persistent_score_player50(turn, opponent);

  return PolicyObservation50{
      own_hand,
      turn.floor,
      own_captured,
      opponent_captured,
      pending_public,
      unseen,
      own_shaken,
      opponent_shaken,
      turn.ppuk_months,
      own_ppuk,
      opponent_ppuk,
      turn.bonus2_ppuk_months,
      turn.bonus3_ppuk_months,
      state.pending_chongtong_months,
      turn.turn_index,
      static_cast<std::uint8_t>(policy_decision_kind50(state)),
      static_cast<std::uint8_t>(core::card_count(turn.stock)),
      go_count_for_player50(state.special.game, player),
      go_count_for_player50(state.special.game, opponent),
      last_go_base_score_for_player50(state.special.game, player),
      last_go_base_score_for_player50(state.special.game, opponent),
      own_score.total_points,
      opponent_score.total_points,
      bombs50(state.special, player),
      bombs50(state.special, opponent),
      credits50(state.special, player),
      credits50(state.special, opponent),
      ppuk_count50(state, player),
      ppuk_count50(state, opponent),
      ppuk_streak50(state, player),
      ppuk_streak50(state, opponent),
      static_cast<std::uint8_t>(
          persistent_gukjin_role50(turn, player) == PersistentGukjinRole50::kDoublePi),
      static_cast<std::uint8_t>(
          persistent_gukjin_role50(turn, opponent) == PersistentGukjinRole50::kDoublePi)};
}

CUGO_HD inline PolicyPacket50 make_policy_packet50(
    const TerminalGameState50& state) noexcept {
  return PolicyPacket50{pack_policy_observation50(state),
                        legal_primary_actions50(state)};
}

CUGO_HD inline bool equal_policy_observation50(
    const PolicyObservation50& a,
    const PolicyObservation50& b) noexcept {
  return a.own_hand == b.own_hand && a.floor == b.floor &&
         a.own_captured == b.own_captured &&
         a.opponent_captured == b.opponent_captured &&
         a.pending_public == b.pending_public && a.unseen == b.unseen &&
         a.own_shaken_months == b.own_shaken_months &&
         a.opponent_shaken_months == b.opponent_shaken_months &&
         a.ppuk_months == b.ppuk_months &&
         a.own_ppuk_months == b.own_ppuk_months &&
         a.opponent_ppuk_months == b.opponent_ppuk_months &&
         a.bonus2_ppuk_months == b.bonus2_ppuk_months &&
         a.bonus3_ppuk_months == b.bonus3_ppuk_months &&
         a.pending_chongtong_months == b.pending_chongtong_months &&
         a.turn_index == b.turn_index &&
         a.decision_kind == b.decision_kind && a.stock_count == b.stock_count &&
         a.own_go_count == b.own_go_count &&
         a.opponent_go_count == b.opponent_go_count &&
         a.own_last_go_base_score == b.own_last_go_base_score &&
         a.opponent_last_go_base_score == b.opponent_last_go_base_score &&
         a.own_base_score == b.own_base_score &&
         a.opponent_base_score == b.opponent_base_score &&
         a.own_bombs == b.own_bombs && a.opponent_bombs == b.opponent_bombs &&
         a.own_credits == b.own_credits &&
         a.opponent_credits == b.opponent_credits &&
         a.own_ppuk_count == b.own_ppuk_count &&
         a.opponent_ppuk_count == b.opponent_ppuk_count &&
         a.own_ppuk_streak == b.own_ppuk_streak &&
         a.opponent_ppuk_streak == b.opponent_ppuk_streak &&
         a.own_gukjin_double_pi == b.own_gukjin_double_pi &&
         a.opponent_gukjin_double_pi == b.opponent_gukjin_double_pi;
}

CUGO_HD inline bool equal_policy_packet50(const PolicyPacket50& a,
                                           const PolicyPacket50& b) noexcept {
  return equal_policy_observation50(a.observation, b.observation) &&
         a.legal_actions.lo == b.legal_actions.lo &&
         a.legal_actions.hi == b.legal_actions.hi;
}

}  // namespace cugo::game

#undef CUGO_HD
