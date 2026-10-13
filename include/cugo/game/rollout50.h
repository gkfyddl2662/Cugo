#pragma once

#include <cstdint>

#include "cugo/game/settlement50.h"

#if defined(__CUDACC__)
#define CUGO_HD __host__ __device__
#else
#define CUGO_HD
#endif

namespace cugo::game {

using core::CardId;
using core::CardMask;

enum class ActionKind50 : std::uint8_t {
  kPlayCard = 0,
  kPlayBonus = 1,
  kShake = 2,
  kBomb = 3,
  kGrenade = 4,
  kBombCredit = 5,
  kGo = 6,
  kStop = 7,
  kChongtongWin = 8,
  kChongtongContinue = 9,
  kGukjinPi = 10,
  kGukjinAnimal = 11,
};

enum class ActionStatus50 : std::uint8_t {
  kOk = 0,
  kWrongPhase = 1,
  kInvalidAction = 2,
  kNotLegal = 3,
  kStockEmpty = 4,
  kResolveChoiceRequired = 5,
  kInvalidResolveChoice = 6,
  kPiSelectionRequired = 7,
  kInvalidPiSelection = 8,
  kGameFinished = 9,
};

struct Action50 {
  ActionKind50 kind;
  CardId card;
  std::uint8_t month;
  Resolve50Choices resolve_choices;
  PiTransferSelection pi_selection;
};

inline constexpr Action50 kInvalidAction50{
    ActionKind50::kPlayCard,
    core::kInvalidCard,
    0xffu,
    kNoResolve50Choices,
    kNoPiTransferSelection};

struct ActionResult50 {
  ActionStatus50 status;
  ActionKind50 kind;
  std::uint8_t actor;
  std::uint8_t turn_completed;
  std::uint8_t decision_opened;
  std::uint8_t terminal_opened;
};

enum class RolloutEnd50 : std::uint8_t {
  kTerminal = 0,
  kNagari = 1,
  kStalled = 2,
  kActionError = 3,
  kMaxActions = 4,
};

struct RolloutResult50 {
  RolloutEnd50 end;
  ActionStatus50 action_status;
  TerminalReason50 terminal_reason;
  std::uint8_t winner;
  std::uint8_t next_round_multiplier;
  std::uint16_t actions;
  std::uint16_t turns;
  std::uint16_t bonus_actions;
  std::uint16_t special_actions;
  std::uint16_t go_actions;
  std::uint64_t final_points;
  std::int64_t reward0;
};

struct RolloutDigest50 {
  std::uint64_t final_points;
  std::int64_t reward0;
  std::uint16_t actions;
  std::uint16_t turns;
  std::uint16_t bonus_actions;
  std::uint16_t special_actions;
  std::uint16_t go_actions;
  std::uint8_t end;
  std::uint8_t action_status;
  std::uint8_t terminal_reason;
  std::uint8_t winner;
  std::uint8_t next_round_multiplier;
};

CUGO_HD inline Action50 make_action50(ActionKind50 kind) noexcept {
  return Action50{kind, core::kInvalidCard, 0xffu,
                  kNoResolve50Choices, kNoPiTransferSelection};
}

CUGO_HD inline std::uint8_t first_month50(std::uint16_t months) noexcept {
  for (std::uint8_t month = 0; month < core::kMonthCount; ++month) {
    if ((months & static_cast<std::uint16_t>(std::uint16_t{1} << month)) != 0)
      return month;
  }
  return 0xffu;
}

CUGO_HD inline PiTransferSelection canonical_pi_selection50(
    const TurnState50& state,
    std::uint8_t stealing_actor,
    std::uint8_t requested_cards) noexcept {
  const std::uint8_t victim = static_cast<std::uint8_t>((stealing_actor & 1u) ^ 1u);
  const CardMask victim_cards = captured_for_player50_persistent(state, victim);
  const CardMask candidates =
      pi_card_mask(victim_cards, persistent_score_options_for_player50(state, victim));
  const std::uint8_t available =
      static_cast<std::uint8_t>(core::card_count(candidates));
  if (requested_cards == 0 || available <= requested_cards) {
    return kNoPiTransferSelection;
  }
  return PiTransferSelection{
      hangame_pi_steal_selection50(state, stealing_actor, requested_cards)};
}

CUGO_HD inline void canonical_choice_step50(CardMask& floor,
                                             CardId card,
                                             CardId& choice) noexcept {
  const CardMask bit = core::card_bit(card);
  const CardMask matches = core::matching_month_cards(floor, card);
  const int count = core::card_count(matches);
  if (count == 0) {
    floor |= bit;
  } else if (count == 1) {
    floor &= ~matches;
  } else if (count == 2) {
    choice = core::first_card(matches);
    floor &= ~core::card_bit(choice);
  } else {
    floor &= ~matches;
  }
}

CUGO_HD inline Resolve50Choices canonical_resolve_choices50(
    const TurnState50& state) noexcept {
  Resolve50Choices choices = kNoResolve50Choices;
  if (state.phase != Turn50Phase::kResolve ||
      !core::is_standard_card(state.pending_played) ||
      !core::is_standard_card(state.pending_drawn)) {
    return choices;
  }

  const CardId played = state.pending_played;
  const CardId drawn = state.pending_drawn;
  CardMask floor = standard_floor50(state);
  const CardMask played_matches = core::matching_month_cards(floor, played);
  const int played_count = core::card_count(played_matches);
  const bool same_month = core::card_month(played) == core::card_month(drawn);
  const bool last_turn = is_last_turn50(state);

  if (same_month && played_count == 1 && !last_turn) return choices;
  if (same_month && (played_count == 0 || played_count == 2)) return choices;

  canonical_choice_step50(floor, played, choices.played_match);
  canonical_choice_step50(floor, drawn, choices.drawn_match);
  return choices;
}

CUGO_HD inline ActionStatus50 map_turn_status50(Turn50Status status) noexcept {
  if (status == Turn50Status::kOk) return ActionStatus50::kOk;
  if (status == Turn50Status::kStockEmpty) return ActionStatus50::kStockEmpty;
  if (status == Turn50Status::kWrongPhase) return ActionStatus50::kWrongPhase;
  return ActionStatus50::kNotLegal;
}

CUGO_HD inline ActionStatus50 map_resolve_status50(
    Resolve50Status status) noexcept {
  if (status == Resolve50Status::kOk) return ActionStatus50::kOk;
  if (status == Resolve50Status::kChoiceRequired)
    return ActionStatus50::kResolveChoiceRequired;
  if (status == Resolve50Status::kInvalidChoice)
    return ActionStatus50::kInvalidResolveChoice;
  if (status == Resolve50Status::kWrongPhase)
    return ActionStatus50::kWrongPhase;
  return ActionStatus50::kNotLegal;
}

CUGO_HD inline ActionStatus50 map_pi_status50(PiTransferStatus status) noexcept {
  if (status == PiTransferStatus::kOk) return ActionStatus50::kOk;
  if (status == PiTransferStatus::kSelectionRequired)
    return ActionStatus50::kPiSelectionRequired;
  return ActionStatus50::kInvalidPiSelection;
}

CUGO_HD inline ActionStatus50 map_special_status50(
    SpecialStatus50 status) noexcept {
  switch (status) {
    case SpecialStatus50::kOk: return ActionStatus50::kOk;
    case SpecialStatus50::kWrongPhase: return ActionStatus50::kWrongPhase;
    case SpecialStatus50::kStockEmpty: return ActionStatus50::kStockEmpty;
    case SpecialStatus50::kResolveChoiceRequired:
      return ActionStatus50::kResolveChoiceRequired;
    case SpecialStatus50::kInvalidResolveChoice:
      return ActionStatus50::kInvalidResolveChoice;
    case SpecialStatus50::kPiSelectionRequired:
      return ActionStatus50::kPiSelectionRequired;
    case SpecialStatus50::kInvalidPiSelection:
      return ActionStatus50::kInvalidPiSelection;
    default: return ActionStatus50::kNotLegal;
  }
}

CUGO_HD inline ActionResult50 action_error50(Action50 action,
                                              ActionStatus50 status,
                                              std::uint8_t actor) noexcept {
  return ActionResult50{status, action.kind, actor, 0, 0, 0};
}

CUGO_HD inline ActionResult50 apply_action50(TerminalGameState50& state,
                                              Action50 action) noexcept {
  const std::uint8_t actor = state.special.game.turn.actor;
  if (terminal50_is_finished(state)) {
    return action_error50(action, ActionStatus50::kGameFinished, actor);
  }

  TerminalGameState50 next = state;

  if (action.kind == ActionKind50::kChongtongWin ||
      action.kind == ActionKind50::kChongtongContinue) {
    const ChongtongAction50 chongtong =
        action.kind == ActionKind50::kChongtongWin
            ? ChongtongAction50::kWin
            : ChongtongAction50::kContinue;
    const TerminalStatus50 status = apply_chongtong_action50(next, chongtong);
    if (status != TerminalStatus50::kOk) {
      return action_error50(action, ActionStatus50::kNotLegal, actor);
    }
    const bool terminal = terminal50_is_finished(next);
    state = next;
    return ActionResult50{ActionStatus50::kOk, action.kind, actor, 0, 0,
                          static_cast<std::uint8_t>(terminal)};
  }

  if (action.kind == ActionKind50::kGo || action.kind == ActionKind50::kStop) {
    const GoStopAction50 go_stop = action.kind == ActionKind50::kGo
        ? GoStopAction50::kGo : GoStopAction50::kStop;
    const GoStopStatus50 status = apply_terminal_go_stop_decision50(next, go_stop);
    if (status != GoStopStatus50::kOk) {
      return action_error50(action, ActionStatus50::kNotLegal, actor);
    }
    const bool terminal = terminal50_is_finished(next);
    state = next;
    return ActionResult50{ActionStatus50::kOk, action.kind, actor, 0, 0,
                          static_cast<std::uint8_t>(terminal)};
  }

  if (action.kind == ActionKind50::kGukjinPi ||
      action.kind == ActionKind50::kGukjinAnimal) {
    const GoStopStatus50 status = apply_terminal_gukjin_decision50(
        next, action.kind == ActionKind50::kGukjinPi);
    if (status != GoStopStatus50::kOk) {
      return action_error50(action, ActionStatus50::kNotLegal, actor);
    }
    const bool decision = game50_has_pending_choice(next.special.game);
    const bool terminal = terminal50_is_finished(next);
    state = next;
    return ActionResult50{ActionStatus50::kOk, action.kind, actor, 0,
                          static_cast<std::uint8_t>(decision),
                          static_cast<std::uint8_t>(terminal)};
  }

  if (terminal50_has_chongtong_choice(next) ||
      game50_has_pending_choice(next.special.game)) {
    return action_error50(action, ActionStatus50::kWrongPhase, actor);
  }

  if (action.kind == ActionKind50::kPlayBonus) {
    const TerminalBonus50Result result = play_bonus_for_terminal50_with_pi_transfer(
        next, action.card, action.pi_selection);
    ActionStatus50 status = map_turn_status50(result.bonus_result.play.status);
    if (status == ActionStatus50::kOk)
      status = map_pi_status50(result.bonus_result.pi_transfer.status);
    if (status != ActionStatus50::kOk)
      return action_error50(action, status, actor);
    state = next;
    return ActionResult50{ActionStatus50::kOk, action.kind, actor, 0, 0, 0};
  }

  if (action.kind == ActionKind50::kBomb ||
      action.kind == ActionKind50::kGrenade ||
      action.kind == ActionKind50::kBombCredit) {
    const SpecialKind50 kind = action.kind == ActionKind50::kBomb
        ? SpecialKind50::kBomb
        : action.kind == ActionKind50::kGrenade
            ? SpecialKind50::kGrenade : SpecialKind50::kBombCredit;
    const SpecialAction50 special{kind, action.card, action.month,
                                  action.resolve_choices.drawn_match,
                                  action.pi_selection};
    const TerminalSpecial50Result result = apply_terminal_special_action50(next, special);
    const ActionStatus50 status = map_special_status50(result.special_result.status);
    if (status != ActionStatus50::kOk)
      return action_error50(action, status, actor);
    const bool decision = game50_has_pending_choice(next.special.game);
    const bool terminal = terminal50_is_finished(next);
    state = next;
    return ActionResult50{ActionStatus50::kOk, action.kind, actor, 1,
                          static_cast<std::uint8_t>(decision),
                          static_cast<std::uint8_t>(terminal)};
  }

  if (action.kind == ActionKind50::kShake) {
    const SpecialAction50 shake{SpecialKind50::kShake, action.card, action.month,
                                core::kInvalidCard, kNoPiTransferSelection};
    const TerminalSpecial50Result started = apply_terminal_special_action50(next, shake);
    ActionStatus50 status = map_special_status50(started.special_result.status);
    if (status != ActionStatus50::kOk)
      return action_error50(action, status, actor);
    status = map_turn_status50(draw_for_terminal50(next));
    if (status != ActionStatus50::kOk)
      return action_error50(action, status, actor);
    const TerminalResolve50Result resolved = resolve_terminal_turn50_with_pi_transfer(
        next, action.resolve_choices, action.pi_selection);
    status = map_resolve_status50(resolved.game_result.turn_result.resolve.status);
    if (status == ActionStatus50::kOk)
      status = map_pi_status50(resolved.game_result.turn_result.pi_transfer.status);
    if (status != ActionStatus50::kOk)
      return action_error50(action, status, actor);
    const bool decision = game50_has_pending_choice(next.special.game);
    const bool terminal = terminal50_is_finished(next);
    state = next;
    return ActionResult50{ActionStatus50::kOk, action.kind, actor, 1,
                          static_cast<std::uint8_t>(decision),
                          static_cast<std::uint8_t>(terminal)};
  }

  if (action.kind == ActionKind50::kPlayCard) {
    // A card of a bomb/grenade month can only be played as that bomb/grenade.
    if (core::is_standard_card(action.card) &&
        (forced_special_months50(next.special) &
         static_cast<std::uint16_t>(std::uint16_t{1} << core::card_month(action.card))) != 0) {
      return action_error50(action, ActionStatus50::kNotLegal, actor);
    }
    ActionStatus50 status = map_turn_status50(begin_regular_play_terminal50(next, action.card));
    if (status != ActionStatus50::kOk)
      return action_error50(action, status, actor);
    status = map_turn_status50(draw_for_terminal50(next));
    if (status != ActionStatus50::kOk)
      return action_error50(action, status, actor);
    const TerminalResolve50Result resolved = resolve_terminal_turn50_with_pi_transfer(
        next, action.resolve_choices, action.pi_selection);
    status = map_resolve_status50(resolved.game_result.turn_result.resolve.status);
    if (status == ActionStatus50::kOk)
      status = map_pi_status50(resolved.game_result.turn_result.pi_transfer.status);
    if (status != ActionStatus50::kOk)
      return action_error50(action, status, actor);
    const bool decision = game50_has_pending_choice(next.special.game);
    const bool terminal = terminal50_is_finished(next);
    state = next;
    return ActionResult50{ActionStatus50::kOk, action.kind, actor, 1,
                          static_cast<std::uint8_t>(decision),
                          static_cast<std::uint8_t>(terminal)};
  }

  return action_error50(action, ActionStatus50::kInvalidAction, actor);
}

CUGO_HD inline ActionStatus50 canonicalize_regular_action50(
    const TerminalGameState50& state,
    Action50& action,
    bool shake) noexcept {
  TerminalGameState50 temp = state;
  const std::uint8_t actor = temp.special.game.turn.actor;
  if (shake) {
    const SpecialAction50 special{SpecialKind50::kShake, action.card, action.month,
                                  core::kInvalidCard, kNoPiTransferSelection};
    const TerminalSpecial50Result started = apply_terminal_special_action50(temp, special);
    const ActionStatus50 special_status = map_special_status50(started.special_result.status);
    if (special_status != ActionStatus50::kOk) return special_status;
  } else {
    const ActionStatus50 play_status =
        map_turn_status50(begin_regular_play_terminal50(temp, action.card));
    if (play_status != ActionStatus50::kOk) return play_status;
  }
  const ActionStatus50 draw_status = map_turn_status50(draw_for_terminal50(temp));
  if (draw_status != ActionStatus50::kOk) return draw_status;
  action.resolve_choices = canonical_resolve_choices50(temp.special.game.turn);
  TurnState50 resolved = temp.special.game.turn;
  const Resolve50Result rr = resolve_turn50(resolved, action.resolve_choices);
  const ActionStatus50 resolve_status = map_resolve_status50(rr.status);
  if (resolve_status != ActionStatus50::kOk) return resolve_status;
  action.pi_selection = canonical_pi_selection50(
      temp.special.game.turn, actor, resolve_pi_steal_card_count(rr));
  return ActionStatus50::kOk;
}

CUGO_HD inline ActionStatus50 canonicalize_bonus_action50(
    const TerminalGameState50& state,
    Action50& action) noexcept {
  TerminalGameState50 temp = state;
  const std::uint8_t actor = temp.special.game.turn.actor;
  const TerminalBonus50Result probe = play_bonus_for_terminal50_with_pi_transfer(
      temp, action.card, kNoPiTransferSelection);
  if (probe.bonus_result.play.status != Turn50Status::kOk)
    return map_turn_status50(probe.bonus_result.play.status);
  if (probe.bonus_result.pi_transfer.status == PiTransferStatus::kSelectionRequired) {
    action.pi_selection = canonical_pi_selection50(
        state.special.game.turn, actor,
        probe.bonus_result.pi_transfer.requested_cards);
    return ActionStatus50::kOk;
  }
  return map_pi_status50(probe.bonus_result.pi_transfer.status);
}

CUGO_HD inline ActionStatus50 canonicalize_special_action50(
    const TerminalGameState50& state,
    Action50& action) noexcept {
  const std::uint8_t actor = state.special.game.turn.actor;
  const SpecialKind50 kind = action.kind == ActionKind50::kBomb
      ? SpecialKind50::kBomb
      : action.kind == ActionKind50::kGrenade
          ? SpecialKind50::kGrenade : SpecialKind50::kBombCredit;

  TerminalGameState50 temp = state;
  SpecialAction50 special{kind, core::kInvalidCard, action.month,
                          core::kInvalidCard, kNoPiTransferSelection};
  TerminalSpecial50Result result = apply_terminal_special_action50(temp, special);
  if (result.special_result.status == SpecialStatus50::kResolveChoiceRequired) {
    CardMask candidates = standard_floor50(state.special.game.turn);
    bool found = false;
    while (candidates != 0) {
      const CardId candidate = core::pop_first_card(candidates);
      TerminalGameState50 candidate_state = state;
      SpecialAction50 candidate_action{kind, core::kInvalidCard, action.month,
                                       candidate, kNoPiTransferSelection};
      const TerminalSpecial50Result candidate_result =
          apply_terminal_special_action50(candidate_state, candidate_action);
      if (candidate_result.special_result.status != SpecialStatus50::kResolveChoiceRequired &&
          candidate_result.special_result.status != SpecialStatus50::kInvalidResolveChoice) {
        action.resolve_choices.drawn_match = candidate;
        result = candidate_result;
        found = true;
        break;
      }
    }
    if (!found) return ActionStatus50::kResolveChoiceRequired;
  }

  if (result.special_result.status == SpecialStatus50::kPiSelectionRequired) {
    action.pi_selection = canonical_pi_selection50(
        state.special.game.turn, actor,
        result.special_result.pi_transfer.requested_cards);
    return ActionStatus50::kOk;
  }
  return map_special_status50(result.special_result.status);
}

struct CanonicalAction50 {
  ActionStatus50 status;
  Action50 action;
};

CUGO_HD inline CanonicalAction50 canonical_action50(
    const TerminalGameState50& state) noexcept {
  if (terminal50_is_finished(state))
    return CanonicalAction50{ActionStatus50::kGameFinished, kInvalidAction50};

  if (terminal50_has_chongtong_choice(state)) {
    Action50 action = make_action50(ActionKind50::kChongtongWin);
    return CanonicalAction50{ActionStatus50::kOk, action};
  }

  if (game50_has_gukjin_decision(state.special.game)) {
    // Hangame's AI always turns Gukjin into double pi when asked.
    return CanonicalAction50{ActionStatus50::kOk, make_action50(ActionKind50::kGukjinPi)};
  }

  if (game50_has_pending_decision(state.special.game)) {
    const std::uint8_t actor = state.special.game.decision_actor;
    const std::uint8_t go_count = go_count_for_player50(state.special.game, actor);
    Action50 action = make_action50(
        go_count == 0 ? ActionKind50::kGo : ActionKind50::kStop);
    return CanonicalAction50{ActionStatus50::kOk, action};
  }

  if (state.special.game.turn.phase != Turn50Phase::kPlay)
    return CanonicalAction50{ActionStatus50::kWrongPhase, kInvalidAction50};

  const std::uint8_t actor = state.special.game.turn.actor;
  const CardMask hand = active_hand50(state.special.game.turn);

  if (credits50(state.special, actor) != 0) {
    Action50 action = make_action50(ActionKind50::kBombCredit);
    const ActionStatus50 status = canonicalize_special_action50(state, action);
    return CanonicalAction50{status, action};
  }

  const CardMask bonuses = hand & core::kBonusCardMask;
  if (bonuses != 0) {
    Action50 action = make_action50(ActionKind50::kPlayBonus);
    action.card = core::first_card(bonuses);
    const ActionStatus50 status = canonicalize_bonus_action50(state, action);
    return CanonicalAction50{status, action};
  }

  const std::uint16_t bombs = legal_bomb_months50(state.special);
  if (bombs != 0) {
    Action50 action = make_action50(ActionKind50::kBomb);
    action.month = first_month50(bombs);
    const ActionStatus50 status = canonicalize_special_action50(state, action);
    return CanonicalAction50{status, action};
  }

  const std::uint16_t grenades = legal_grenade_months50(state.special);
  if (grenades != 0) {
    Action50 action = make_action50(ActionKind50::kGrenade);
    action.month = first_month50(grenades);
    const ActionStatus50 status = canonicalize_special_action50(state, action);
    return CanonicalAction50{status, action};
  }

  const std::uint16_t shakes = legal_shake_months50(state.special);
  if (shakes != 0) {
    Action50 action = make_action50(ActionKind50::kShake);
    action.month = first_month50(shakes);
    action.card = core::first_card(
        month_cards50(hand & core::kStandardDeckMask, action.month));
    const ActionStatus50 status = canonicalize_regular_action50(state, action, true);
    return CanonicalAction50{status, action};
  }

  const CardMask standard = hand & core::kStandardDeckMask;
  if (standard != 0) {
    Action50 action = make_action50(ActionKind50::kPlayCard);
    action.card = core::first_card(standard);
    const ActionStatus50 status = canonicalize_regular_action50(state, action, false);
    return CanonicalAction50{status, action};
  }

  return CanonicalAction50{ActionStatus50::kNotLegal, kInvalidAction50};
}

CUGO_HD inline bool rollout_hands_exhausted50(
    const TerminalGameState50& state) noexcept {
  return state.special.game.turn.phase == Turn50Phase::kPlay &&
         state.special.game.turn.hand0 == 0 &&
         state.special.game.turn.hand1 == 0 &&
         state.special.credits0 == 0 && state.special.credits1 == 0;
}

CUGO_HD inline RolloutResult50 finish_terminal_rollout50(
    const TerminalGameState50& state,
    std::uint16_t actions,
    std::uint16_t bonus_actions,
    std::uint16_t special_actions,
    std::uint16_t go_actions) noexcept {
  const Settlement50 settlement = settle_terminal50(state);
  if (settlement.status != SettlementStatus50::kOk) {
    return RolloutResult50{RolloutEnd50::kActionError,
                           ActionStatus50::kInvalidAction,
                           state.terminal_reason,
                           state.terminal_winner,
                           1,
                           actions,
                           state.special.game.turn.turn_index,
                           bonus_actions,
                           special_actions,
                           go_actions,
                           0,
                           0};
  }
  return RolloutResult50{RolloutEnd50::kTerminal,
                         ActionStatus50::kOk,
                         state.terminal_reason,
                         settlement.winner,
                         1,
                         actions,
                         state.special.game.turn.turn_index,
                         bonus_actions,
                         special_actions,
                         go_actions,
                         settlement.final_points,
                         settlement_reward50(settlement, 0)};
}

CUGO_HD inline RolloutResult50 rollout_canonical50(
    TerminalGameState50& state,
    std::uint16_t max_actions = 128) noexcept {
  std::uint16_t actions = 0;
  std::uint16_t bonus_actions = 0;
  std::uint16_t special_actions = 0;
  std::uint16_t go_actions = 0;

  while (actions < max_actions) {
    if (terminal50_is_finished(state)) {
      return finish_terminal_rollout50(state, actions, bonus_actions,
                                       special_actions, go_actions);
    }

    if (!terminal50_has_chongtong_choice(state) &&
        !game50_has_pending_choice(state.special.game) &&
        rollout_hands_exhausted50(state)) {
      return RolloutResult50{RolloutEnd50::kNagari,
                             ActionStatus50::kOk,
                             TerminalReason50::kNone,
                             kNoTerminal50Player,
                             2,
                             actions,
                             state.special.game.turn.turn_index,
                             bonus_actions,
                             special_actions,
                             go_actions,
                             0,
                             nagari_reward50(state, 0)};
    }

    const CanonicalAction50 canonical = canonical_action50(state);
    if (canonical.status != ActionStatus50::kOk) {
      return RolloutResult50{RolloutEnd50::kStalled,
                             canonical.status,
                             TerminalReason50::kNone,
                             kNoTerminal50Player,
                             1,
                             actions,
                             state.special.game.turn.turn_index,
                             bonus_actions,
                             special_actions,
                             go_actions,
                             0,
                             0};
    }

    const Action50 action = canonical.action;
    const ActionResult50 applied = apply_action50(state, action);
    if (applied.status != ActionStatus50::kOk) {
      return RolloutResult50{RolloutEnd50::kActionError,
                             applied.status,
                             TerminalReason50::kNone,
                             kNoTerminal50Player,
                             1,
                             actions,
                             state.special.game.turn.turn_index,
                             bonus_actions,
                             special_actions,
                             go_actions,
                             0,
                             0};
    }

    ++actions;
    if (action.kind == ActionKind50::kPlayBonus) ++bonus_actions;
    if (action.kind == ActionKind50::kShake ||
        action.kind == ActionKind50::kBomb ||
        action.kind == ActionKind50::kGrenade ||
        action.kind == ActionKind50::kBombCredit) {
      ++special_actions;
    }
    if (action.kind == ActionKind50::kGo) ++go_actions;
  }

  return RolloutResult50{RolloutEnd50::kMaxActions,
                         ActionStatus50::kOk,
                         TerminalReason50::kNone,
                         kNoTerminal50Player,
                         1,
                         actions,
                         state.special.game.turn.turn_index,
                         bonus_actions,
                         special_actions,
                         go_actions,
                         0,
                         0};
}

CUGO_HD inline RolloutResult50 rollout_canonical50(
    std::uint64_t seed,
    std::uint8_t first_player = 0,
    std::uint16_t max_actions = 128) noexcept {
  TerminalGameState50 state =
      make_terminal_game_state50(deal_shin_matgo_50(seed), first_player);
  return rollout_canonical50(state, max_actions);
}

CUGO_HD inline RolloutDigest50 rollout_digest50(
    const RolloutResult50& result) noexcept {
  return RolloutDigest50{result.final_points,
                         result.reward0,
                         result.actions,
                         result.turns,
                         result.bonus_actions,
                         result.special_actions,
                         result.go_actions,
                         static_cast<std::uint8_t>(result.end),
                         static_cast<std::uint8_t>(result.action_status),
                         static_cast<std::uint8_t>(result.terminal_reason),
                         result.winner,
                         result.next_round_multiplier};
}

CUGO_HD inline bool equal_rollout_digest50(const RolloutDigest50& a,
                                            const RolloutDigest50& b) noexcept {
  return a.final_points == b.final_points && a.reward0 == b.reward0 &&
         a.actions == b.actions && a.turns == b.turns &&
         a.bonus_actions == b.bonus_actions &&
         a.special_actions == b.special_actions &&
         a.go_actions == b.go_actions && a.end == b.end &&
         a.action_status == b.action_status &&
         a.terminal_reason == b.terminal_reason && a.winner == b.winner &&
         a.next_round_multiplier == b.next_round_multiplier;
}

}  // namespace cugo::game

#undef CUGO_HD
