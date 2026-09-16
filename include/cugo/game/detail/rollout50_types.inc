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

CUGO_HD inline CardMask take_first_cards50(CardMask cards,
                                            std::uint8_t count) noexcept {
  CardMask out = 0;
  while (cards != 0 && count != 0) {
    const CardId card = core::pop_first_card(cards);
    out |= core::card_bit(card);
    --count;
  }
  return out;
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
  return PiTransferSelection{take_first_cards50(candidates, requested_cards)};
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
  const bool final_stock_flip = state.stock == 0;

  if (same_month && played_count == 1 && !final_stock_flip) return choices;
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
