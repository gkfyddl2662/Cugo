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
