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

  if (terminal50_has_chongtong_choice(next) ||
      game50_has_pending_decision(next.special.game)) {
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
    const bool decision = game50_has_pending_decision(next.special.game);
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
    const bool decision = game50_has_pending_decision(next.special.game);
    const bool terminal = terminal50_is_finished(next);
    state = next;
    return ActionResult50{ActionStatus50::kOk, action.kind, actor, 1,
                          static_cast<std::uint8_t>(decision),
                          static_cast<std::uint8_t>(terminal)};
  }

  if (action.kind == ActionKind50::kPlayCard) {
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
    const bool decision = game50_has_pending_decision(next.special.game);
    const bool terminal = terminal50_is_finished(next);
    state = next;
    return ActionResult50{ActionStatus50::kOk, action.kind, actor, 1,
                          static_cast<std::uint8_t>(decision),
                          static_cast<std::uint8_t>(terminal)};
  }

  return action_error50(action, ActionStatus50::kInvalidAction, actor);
}
