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
