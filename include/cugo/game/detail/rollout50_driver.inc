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
        !game50_has_pending_decision(state.special.game) &&
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
                             0};
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
