#pragma once

#include <cstdint>

#include "cugo/game/policy50.h"

#if defined(__CUDACC__)
#define CUGO_HD __host__ __device__
#else
#define CUGO_HD
#endif

namespace cugo::game {

enum class PostChanceDecision50 : std::uint8_t {
  kNone = 0,
  kResolvePlayed = 4,
  kResolveDrawn = 5,
  kPiCard = 6,
};

struct PostChancePending50 {
  Action50 action;
  CardMask selected_pi;
  std::uint16_t primary_action;
  std::uint8_t actor;
  std::uint8_t active;
};

struct PostChancePacket50 {
  PolicyObservation50 observation;
  CardMask legal_cards;
  CardMask selected_pi;
  std::uint16_t primary_action;
  CardId revealed_card;
  std::uint8_t requested_pi;
  std::uint8_t decision_player;
  PostChanceDecision50 decision;
};

struct PostChanceStep50 {
  ActionStatus50 status;
  std::uint8_t action_committed;
  PostChancePacket50 packet;
  ActionResult50 applied;
};

struct PostChanceRollout50 {
  RolloutResult50 result;
  std::uint16_t resolve_choices;
  std::uint16_t pi_choices;
};

struct ResolveChoiceProbe50 {
  Resolve50Status status;
  PostChanceDecision50 decision;
  CardMask legal_cards;
};

struct PostChanceProbe50 {
  ActionStatus50 status;
  PostChanceDecision50 decision;
  CardMask legal_cards;
  CardId revealed_card;
  std::uint8_t requested_pi;
};

static_assert(sizeof(PostChancePacket50) <= 128,
              "post-chance packet should stay cache friendly");

CUGO_HD inline PostChancePacket50 empty_postchance_packet50() noexcept {
  return PostChancePacket50{};
}

CUGO_HD inline ActionResult50 empty_postchance_action_result50(
    ActionKind50 kind,
    std::uint8_t actor) noexcept {
  return ActionResult50{ActionStatus50::kOk, kind, actor, 0, 0, 0};
}

CUGO_HD inline ActionStatus50 decode_primary_action50_raw(
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
    return ActionStatus50::kOk;
  }
  if (index < kPolicyShakeBegin50) {
    action = make_action50(ActionKind50::kPlayBonus);
    action.card = index == kPolicyPlayBonusBegin50
        ? core::kBonusTwoPi : core::kBonusThreePi;
    return ActionStatus50::kOk;
  }
  if (index < kPolicyBombBegin50) {
    action = make_action50(ActionKind50::kShake);
    action.card = static_cast<CardId>(index - kPolicyShakeBegin50);
    action.month = core::card_month(action.card);
    return ActionStatus50::kOk;
  }
  if (index < kPolicyGrenadeBegin50) {
    action = make_action50(ActionKind50::kBomb);
    action.month = static_cast<std::uint8_t>(index - kPolicyBombBegin50);
    return ActionStatus50::kOk;
  }
  if (index < kPolicyBombCredit50) {
    action = make_action50(ActionKind50::kGrenade);
    action.month = static_cast<std::uint8_t>(index - kPolicyGrenadeBegin50);
    return ActionStatus50::kOk;
  }
  if (index == kPolicyBombCredit50) {
    action = make_action50(ActionKind50::kBombCredit);
    return ActionStatus50::kOk;
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

CUGO_HD inline ResolveChoiceProbe50 probe_resolve_choices50(
    const TurnState50& state,
    Resolve50Choices choices) noexcept {
  if (state.phase != Turn50Phase::kResolve) {
    return ResolveChoiceProbe50{Resolve50Status::kWrongPhase,
                                PostChanceDecision50::kNone, 0};
  }

  const CardId played = state.pending_played;
  const CardId drawn = state.pending_drawn;
  const CardMask played_matches = floor_matches50(state, played);
  const int played_match_count = core::card_count(played_matches);
  const bool same_month = core::card_month(played) == core::card_month(drawn);
  const bool final_stock_flip = state.stock == 0;

  if (same_month && played_match_count == 1 && !final_stock_flip) {
    return ResolveChoiceProbe50{Resolve50Status::kOk,
                                PostChanceDecision50::kNone, 0};
  }
  if (same_month && (played_match_count == 0 || played_match_count == 2)) {
    return ResolveChoiceProbe50{Resolve50Status::kOk,
                                PostChanceDecision50::kNone, 0};
  }

  TurnState50 next = state;
  Resolve50Result result{Resolve50Status::kOk, kResolve50EventNone, 0, 0, 0};

  CardMask candidates = floor_matches50(next, played);
  Resolve50Status status = resolve_card_against_floor50(
      next, played, choices.played_match, result);
  if (status == Resolve50Status::kChoiceRequired) {
    return ResolveChoiceProbe50{status, PostChanceDecision50::kResolvePlayed,
                                candidates};
  }
  if (status != Resolve50Status::kOk) {
    return ResolveChoiceProbe50{status, PostChanceDecision50::kNone, 0};
  }

  candidates = floor_matches50(next, drawn);
  status = resolve_card_against_floor50(
      next, drawn, choices.drawn_match, result);
  if (status == Resolve50Status::kChoiceRequired) {
    return ResolveChoiceProbe50{status, PostChanceDecision50::kResolveDrawn,
                                candidates};
  }
  if (status != Resolve50Status::kOk) {
    return ResolveChoiceProbe50{status, PostChanceDecision50::kNone, 0};
  }
  return ResolveChoiceProbe50{Resolve50Status::kOk,
                              PostChanceDecision50::kNone, 0};
}

CUGO_HD inline CardMask postchance_pi_candidates50(
    const TurnState50& turn,
    std::uint8_t actor) noexcept {
  const std::uint8_t victim = static_cast<std::uint8_t>((actor & 1u) ^ 1u);
  return pi_card_mask(captured_for_player50_persistent(turn, victim),
                      persistent_score_options_for_player50(turn, victim));
}

CUGO_HD inline PostChanceProbe50 probe_pi_choice50(
    const TerminalGameState50& staged,
    const PostChancePending50& pending,
    CardId revealed_card,
    std::uint8_t requested_pi) noexcept {
  const CardMask candidates =
      postchance_pi_candidates50(staged.special.game.turn, pending.actor);
  const std::uint8_t available =
      static_cast<std::uint8_t>(core::card_count(candidates));
  const std::uint8_t selected =
      static_cast<std::uint8_t>(core::card_count(pending.selected_pi));

  if ((pending.selected_pi & ~candidates) != 0) {
    return PostChanceProbe50{ActionStatus50::kInvalidPiSelection,
                             PostChanceDecision50::kNone, 0,
                             revealed_card, requested_pi};
  }

  if (requested_pi == 0 || available <= requested_pi) {
    if (pending.selected_pi != 0) {
      return PostChanceProbe50{ActionStatus50::kInvalidPiSelection,
                               PostChanceDecision50::kNone, 0,
                               revealed_card, requested_pi};
    }
    return PostChanceProbe50{ActionStatus50::kOk,
                             PostChanceDecision50::kNone, 0,
                             revealed_card, requested_pi};
  }

  if (selected > requested_pi) {
    return PostChanceProbe50{ActionStatus50::kInvalidPiSelection,
                             PostChanceDecision50::kNone, 0,
                             revealed_card, requested_pi};
  }
  if (selected < requested_pi) {
    return PostChanceProbe50{ActionStatus50::kOk,
                             PostChanceDecision50::kPiCard,
                             candidates & ~pending.selected_pi,
                             revealed_card, requested_pi};
  }
  return PostChanceProbe50{ActionStatus50::kOk,
                           PostChanceDecision50::kNone, 0,
                           revealed_card, requested_pi};
}

CUGO_HD inline PostChanceProbe50 probe_regular_postchance50(
    const TerminalGameState50& state,
    const PostChancePending50& pending,
    TerminalGameState50& staged,
    bool shake) noexcept {
  staged = state;
  ActionStatus50 status = ActionStatus50::kOk;
  if (shake) {
    const SpecialResult50 started = begin_shake50(staged.special, pending.action.card);
    status = map_special_status50(started.status);
  } else {
    status = map_turn_status50(
        begin_regular_play_terminal50(staged, pending.action.card));
  }
  if (status != ActionStatus50::kOk) {
    return PostChanceProbe50{status, PostChanceDecision50::kNone, 0,
                             core::kInvalidCard, 0};
  }

  status = map_turn_status50(draw_for_terminal50(staged));
  if (status != ActionStatus50::kOk) {
    return PostChanceProbe50{status, PostChanceDecision50::kNone, 0,
                             core::kInvalidCard, 0};
  }

  const CardId revealed = staged.special.game.turn.pending_drawn;
  const ResolveChoiceProbe50 choice = probe_resolve_choices50(
      staged.special.game.turn, pending.action.resolve_choices);
  if (choice.status == Resolve50Status::kChoiceRequired) {
    return PostChanceProbe50{ActionStatus50::kOk, choice.decision,
                             choice.legal_cards, revealed, 0};
  }
  if (choice.status == Resolve50Status::kInvalidChoice) {
    return PostChanceProbe50{ActionStatus50::kInvalidResolveChoice,
                             PostChanceDecision50::kNone, 0, revealed, 0};
  }
  if (choice.status != Resolve50Status::kOk) {
    return PostChanceProbe50{ActionStatus50::kNotLegal,
                             PostChanceDecision50::kNone, 0, revealed, 0};
  }

  TurnState50 resolved = staged.special.game.turn;
  const Resolve50Result rr = resolve_turn50(resolved, pending.action.resolve_choices);
  if (rr.status != Resolve50Status::kOk) {
    return PostChanceProbe50{map_resolve_status50(rr.status),
                             PostChanceDecision50::kNone, 0, revealed, 0};
  }
  staged.special.game.turn = resolved;
  return probe_pi_choice50(staged, pending, revealed,
                           resolve_pi_steal_card_count(rr));
}

CUGO_HD inline PostChanceProbe50 probe_bonus_postchance50(
    const TerminalGameState50& state,
    const PostChancePending50& pending,
    TerminalGameState50& staged) noexcept {
  staged = state;
  const Turn50BonusPlayResult played =
      play_bonus_for_turn50(staged.special.game.turn, pending.action.card);
  const ActionStatus50 status = map_turn_status50(played.status);
  if (status != ActionStatus50::kOk) {
    return PostChanceProbe50{status, PostChanceDecision50::kNone, 0,
                             core::kInvalidCard, 0};
  }
  return probe_pi_choice50(staged, pending, played.replacement,
                           played.pi_steal_count);
}

CUGO_HD inline PostChanceProbe50 probe_special_postchance50(
    const TerminalGameState50& state,
    const PostChancePending50& pending,
    TerminalGameState50& staged) noexcept {
  staged = state;
  SpecialGameState50& special = staged.special;
  TurnState50& turn = special.game.turn;
  const std::uint8_t actor = pending.actor;
  std::uint8_t base_pi_steal = 0;

  if (pending.action.kind == ActionKind50::kBomb ||
      pending.action.kind == ActionKind50::kGrenade) {
    const bool grenade = pending.action.kind == ActionKind50::kGrenade;
    const std::uint8_t month = pending.action.month;
    if (!special_ready50(special) || month >= core::kMonthCount) {
      return PostChanceProbe50{ActionStatus50::kNotLegal,
                               PostChanceDecision50::kNone, 0,
                               core::kInvalidCard, 0};
    }
    const std::uint16_t month_bit = static_cast<std::uint16_t>(
        std::uint16_t{1} << month);
    const std::uint16_t legal = grenade ? legal_grenade_months50(special)
                                        : legal_bomb_months50(special);
    if ((legal & month_bit) == 0) {
      return PostChanceProbe50{ActionStatus50::kNotLegal,
                               PostChanceDecision50::kNone, 0,
                               core::kInvalidCard, 0};
    }

    CardMask& hand = actor == 0 ? turn.hand0 : turn.hand1;
    CardMask& captured = actor == 0 ? turn.captured0 : turn.captured1;
    const CardMask hand_cards = month_cards50(hand, month);
    const CardMask floor_cards = month_cards50(standard_floor50(turn), month);
    hand &= ~hand_cards;
    turn.floor &= ~floor_cards;
    captured |= hand_cards | floor_cards;
    if (actor == 0) {
      special.credits0 = static_cast<std::uint8_t>(
          special.credits0 + (grenade ? 1u : 2u));
      if (!grenade) ++special.bombs0;
    } else {
      special.credits1 = static_cast<std::uint8_t>(
          special.credits1 + (grenade ? 1u : 2u));
      if (!grenade) ++special.bombs1;
    }
    base_pi_steal = 1;
  } else if (pending.action.kind == ActionKind50::kBombCredit) {
    if (!special_ready50(special) || credits50(special, actor) == 0) {
      return PostChanceProbe50{ActionStatus50::kNotLegal,
                               PostChanceDecision50::kNone, 0,
                               core::kInvalidCard, 0};
    }
    if (actor == 0) --special.credits0;
    else --special.credits1;
  } else {
    return PostChanceProbe50{ActionStatus50::kInvalidAction,
                             PostChanceDecision50::kNone, 0,
                             core::kInvalidCard, 0};
  }

  const DrawOnlySpecial50 draw = draw_only_special50(
      turn, pending.action.resolve_choices.drawn_match);
  if (draw.status == SpecialStatus50::kResolveChoiceRequired) {
    turn.pending_drawn = draw.drawn;
    turn.pending_bonus_mask = draw.bonuses;
    return PostChanceProbe50{ActionStatus50::kOk,
                             PostChanceDecision50::kResolveDrawn,
                             floor_matches50(turn, draw.drawn),
                             draw.drawn, 0};
  }
  if (draw.status == SpecialStatus50::kInvalidResolveChoice) {
    return PostChanceProbe50{ActionStatus50::kInvalidResolveChoice,
                             PostChanceDecision50::kNone, 0,
                             draw.drawn, 0};
  }
  if (draw.status != SpecialStatus50::kOk) {
    return PostChanceProbe50{map_special_status50(draw.status),
                             PostChanceDecision50::kNone, 0,
                             draw.drawn, 0};
  }

  const std::uint8_t requested_pi = static_cast<std::uint8_t>(
      base_pi_steal + resolve_pi_steal_card_count(draw.resolve));
  finish_draw_only50(turn);
  return probe_pi_choice50(staged, pending, draw.drawn, requested_pi);
}

CUGO_HD inline PostChanceProbe50 probe_postchance50(
    const TerminalGameState50& state,
    const PostChancePending50& pending,
    TerminalGameState50& staged) noexcept {
  if (!pending.active) {
    staged = state;
    return PostChanceProbe50{ActionStatus50::kWrongPhase,
                             PostChanceDecision50::kNone, 0,
                             core::kInvalidCard, 0};
  }

  switch (pending.action.kind) {
    case ActionKind50::kPlayCard:
      return probe_regular_postchance50(state, pending, staged, false);
    case ActionKind50::kShake:
      return probe_regular_postchance50(state, pending, staged, true);
    case ActionKind50::kPlayBonus:
      return probe_bonus_postchance50(state, pending, staged);
    case ActionKind50::kBomb:
    case ActionKind50::kGrenade:
    case ActionKind50::kBombCredit:
      return probe_special_postchance50(state, pending, staged);
    case ActionKind50::kGo:
    case ActionKind50::kStop:
    case ActionKind50::kChongtongWin:
    case ActionKind50::kChongtongContinue:
      staged = state;
      return PostChanceProbe50{ActionStatus50::kOk,
                               PostChanceDecision50::kNone, 0,
                               core::kInvalidCard, 0};
    default:
      staged = state;
      return PostChanceProbe50{ActionStatus50::kInvalidAction,
                               PostChanceDecision50::kNone, 0,
                               core::kInvalidCard, 0};
  }
}

CUGO_HD inline PolicyObservation50 pack_postchance_observation50(
    const TerminalGameState50& original,
    const TerminalGameState50& staged,
    const PostChancePending50& pending,
    const PostChanceProbe50& probe) noexcept {
  TerminalGameState50 view = staged;
  view.special.game.turn.actor = pending.actor;
  view.special.game.turn.turn_index = original.special.game.turn.turn_index;
  PolicyObservation50 observation = pack_policy_observation50(view);
  observation.decision_kind = static_cast<std::uint8_t>(probe.decision);

  if (core::is_physical_card(probe.revealed_card)) {
    const CardMask bit = core::card_bit(probe.revealed_card);
    const CardMask known = observation.own_hand | observation.floor |
                           observation.own_captured |
                           observation.opponent_captured |
                           observation.pending_public;
    if ((known & bit) == 0) {
      observation.pending_public |= bit;
      observation.unseen &= ~bit;
    }
  }
  return observation;
}

CUGO_HD inline PostChancePacket50 make_postchance_packet50(
    const TerminalGameState50& original,
    const TerminalGameState50& staged,
    const PostChancePending50& pending,
    const PostChanceProbe50& probe) noexcept {
  return PostChancePacket50{
      pack_postchance_observation50(original, staged, pending, probe),
      probe.legal_cards,
      pending.selected_pi,
      pending.primary_action,
      probe.revealed_card,
      probe.requested_pi,
      pending.actor,
      probe.decision};
}

CUGO_HD inline PostChanceStep50 postchance_error_step50(
    const PostChancePending50& pending,
    ActionStatus50 status) noexcept {
  return PostChanceStep50{
      status,
      0,
      empty_postchance_packet50(),
      empty_postchance_action_result50(pending.action.kind, pending.actor)};
}

CUGO_HD inline PostChanceStep50 continue_postchance50(
    TerminalGameState50& state,
    PostChancePending50& pending) noexcept {
  TerminalGameState50 staged{};
  const PostChanceProbe50 probe = probe_postchance50(state, pending, staged);
  if (probe.status != ActionStatus50::kOk) {
    return postchance_error_step50(pending, probe.status);
  }

  if (probe.decision != PostChanceDecision50::kNone) {
    return PostChanceStep50{
        ActionStatus50::kOk,
        0,
        make_postchance_packet50(state, staged, pending, probe),
        empty_postchance_action_result50(pending.action.kind, pending.actor)};
  }

  Action50 action = pending.action;
  if (pending.selected_pi != 0) {
    action.pi_selection = PiTransferSelection{pending.selected_pi};
  }
  const ActionResult50 applied = apply_action50(state, action);
  if (applied.status != ActionStatus50::kOk) {
    return PostChanceStep50{applied.status, 0, empty_postchance_packet50(),
                            applied};
  }
  pending.action = action;
  pending.active = 0;
  return PostChanceStep50{ActionStatus50::kOk, 1,
                          empty_postchance_packet50(), applied};
}

CUGO_HD inline PostChanceStep50 start_postchance_action50(
    TerminalGameState50& state,
    std::uint16_t primary_action,
    PostChancePending50& pending) noexcept {
  Action50 action = kInvalidAction50;
  const ActionStatus50 status =
      decode_primary_action50_raw(state, primary_action, action);
  std::uint8_t actor = policy_decision_player50(state);
  if (actor > 1u) actor = static_cast<std::uint8_t>(state.special.game.turn.actor & 1u);
  pending = PostChancePending50{action, 0, primary_action, actor, 0};
  if (status != ActionStatus50::kOk) {
    return postchance_error_step50(pending, status);
  }
  pending.active = 1;
  return continue_postchance50(state, pending);
}

CUGO_HD inline PostChanceStep50 choose_postchance_card50(
    TerminalGameState50& state,
    PostChancePending50& pending,
    CardId card) noexcept {
  TerminalGameState50 staged{};
  const PostChanceProbe50 probe = probe_postchance50(state, pending, staged);
  if (probe.status != ActionStatus50::kOk) {
    return postchance_error_step50(pending, probe.status);
  }
  if (probe.decision == PostChanceDecision50::kNone) {
    return continue_postchance50(state, pending);
  }
  if (!core::is_physical_card(card) ||
      (probe.legal_cards & core::card_bit(card)) == 0) {
    const ActionStatus50 invalid =
        probe.decision == PostChanceDecision50::kPiCard
            ? ActionStatus50::kInvalidPiSelection
            : ActionStatus50::kInvalidResolveChoice;
    return postchance_error_step50(pending, invalid);
  }

  if (probe.decision == PostChanceDecision50::kResolvePlayed) {
    pending.action.resolve_choices.played_match = card;
  } else if (probe.decision == PostChanceDecision50::kResolveDrawn) {
    pending.action.resolve_choices.drawn_match = card;
  } else if (probe.decision == PostChanceDecision50::kPiCard) {
    pending.selected_pi |= core::card_bit(card);
  }
  return continue_postchance50(state, pending);
}

CUGO_HD inline PostChanceRollout50 finish_postchance_rollout50(
    const RolloutResult50& result,
    std::uint16_t resolve_choices,
    std::uint16_t pi_choices) noexcept {
  return PostChanceRollout50{result, resolve_choices, pi_choices};
}

CUGO_HD inline PostChanceRollout50 rollout_canonical_postchance50(
    TerminalGameState50& state,
    std::uint16_t max_actions = 128) noexcept {
  std::uint16_t actions = 0;
  std::uint16_t bonus_actions = 0;
  std::uint16_t special_actions = 0;
  std::uint16_t go_actions = 0;
  std::uint16_t resolve_choices = 0;
  std::uint16_t pi_choices = 0;

  while (actions < max_actions) {
    if (terminal50_is_finished(state)) {
      return finish_postchance_rollout50(
          finish_terminal_rollout50(state, actions, bonus_actions,
                                    special_actions, go_actions),
          resolve_choices, pi_choices);
    }

    if (!terminal50_has_chongtong_choice(state) &&
        !game50_has_pending_decision(state.special.game) &&
        rollout_hands_exhausted50(state)) {
      const RolloutResult50 result{
          RolloutEnd50::kNagari,
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
          0};
      return finish_postchance_rollout50(result, resolve_choices, pi_choices);
    }

    const CanonicalAction50 canonical = canonical_action50(state);
    if (canonical.status != ActionStatus50::kOk) {
      const RolloutResult50 result{
          RolloutEnd50::kStalled,
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
      return finish_postchance_rollout50(result, resolve_choices, pi_choices);
    }

    const std::uint16_t primary = primary_action_index50(canonical.action);
    if (primary == kInvalidPolicyAction50) {
      const RolloutResult50 result{
          RolloutEnd50::kActionError,
          ActionStatus50::kInvalidAction,
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
      return finish_postchance_rollout50(result, resolve_choices, pi_choices);
    }

    PostChancePending50 pending{};
    PostChanceStep50 step = start_postchance_action50(state, primary, pending);
    const ActionKind50 kind = pending.action.kind;
    while (step.status == ActionStatus50::kOk && !step.action_committed) {
      if (step.packet.legal_cards == 0) break;
      if (step.packet.decision == PostChanceDecision50::kPiCard) ++pi_choices;
      else ++resolve_choices;
      const CardId choice = core::first_card(step.packet.legal_cards);
      step = choose_postchance_card50(state, pending, choice);
    }

    if (step.status != ActionStatus50::kOk || !step.action_committed) {
      const RolloutResult50 result{
          RolloutEnd50::kActionError,
          step.status == ActionStatus50::kOk
              ? ActionStatus50::kInvalidAction : step.status,
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
      return finish_postchance_rollout50(result, resolve_choices, pi_choices);
    }

    ++actions;
    if (kind == ActionKind50::kPlayBonus) ++bonus_actions;
    if (kind == ActionKind50::kShake || kind == ActionKind50::kBomb ||
        kind == ActionKind50::kGrenade || kind == ActionKind50::kBombCredit) {
      ++special_actions;
    }
    if (kind == ActionKind50::kGo) ++go_actions;
  }

  const RolloutResult50 result{
      RolloutEnd50::kMaxActions,
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
  return finish_postchance_rollout50(result, resolve_choices, pi_choices);
}

CUGO_HD inline PostChanceRollout50 rollout_canonical_postchance50(
    std::uint64_t seed,
    std::uint8_t first_player = 0,
    std::uint16_t max_actions = 128) noexcept {
  TerminalGameState50 state =
      make_terminal_game_state50(deal_shin_matgo_50(seed), first_player);
  return rollout_canonical_postchance50(state, max_actions);
}

}  // namespace cugo::game

#undef CUGO_HD
