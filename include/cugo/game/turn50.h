#pragma once

#include <cstddef>
#include <cstdint>

#include "cugo/core/card.h"
#include "cugo/game/bonus.h"
#include "cugo/game/deal.h"

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::game {

using core::CardId;
using core::CardMask;

inline constexpr std::uint16_t kAllMonthBits50 =
    (std::uint16_t{1} << core::kMonthCount) - 1;

enum class Turn50Phase : std::uint8_t {
  kPlay = 0,
  kDraw = 1,
  kResolve = 2,
};

enum class Turn50Status : std::uint8_t {
  kOk = 0,
  kWrongPhase = 1,
  kInvalidCard = 2,
  kCardNotInHand = 3,
  kStockEmpty = 4,
};

enum class Resolve50Status : std::uint8_t {
  kOk = 0,
  kWrongPhase = 1,
  kChoiceRequired = 2,
  kInvalidChoice = 3,
  kUnsupportedLastCardSpecial = 4,
};

enum Resolve50Event : std::uint8_t {
  kResolve50EventNone = 0,
  kResolve50EventPpuk = 1u << 0,
  kResolve50EventJjok = 1u << 1,
  kResolve50EventTtadak = 1u << 2,
  kResolve50EventSweep = 1u << 3,
};

struct Resolve50Choices {
  CardId played_match;
  CardId drawn_match;
};

inline constexpr Resolve50Choices kNoResolve50Choices{core::kInvalidCard,
                                                       core::kInvalidCard};

struct Resolve50Result {
  Resolve50Status status;
  std::uint8_t events;
  std::uint8_t captured_own_ppuk;
  std::uint8_t captured_opponent_ppuk;
  CardMask captured_cards;
};

struct Turn50BonusPlayResult {
  Turn50Status status;
  CardId replacement;
  std::uint8_t pi_steal_count;
};

struct TurnState50 {
  CardMask hand0;
  CardMask hand1;
  CardMask floor;
  CardMask stock;
  CardMask captured0;
  CardMask captured1;
  std::uint64_t rng_state;
  CardMask pending_bonus_mask;
  std::uint16_t ppuk_months;
  std::uint16_t ppuk_owner1_months;
  std::uint16_t bonus2_ppuk_months;
  std::uint16_t bonus3_ppuk_months;
  std::uint16_t turn_index;
  std::uint8_t actor;
  Turn50Phase phase;
  CardId pending_played;
  CardId pending_drawn;
};

struct TurnStateSoA50 {
  CardMask* hand0;
  CardMask* hand1;
  CardMask* floor;
  CardMask* stock;
  CardMask* captured0;
  CardMask* captured1;
  std::uint64_t* rng_state;
  CardMask* pending_bonus_mask;
  std::uint16_t* ppuk_months;
  std::uint16_t* ppuk_owner1_months;
  std::uint16_t* bonus2_ppuk_months;
  std::uint16_t* bonus3_ppuk_months;
  std::uint16_t* turn_index;
  std::uint8_t* actor;
  std::uint8_t* phase;
  CardId* pending_played;
  CardId* pending_drawn;
};

CUGO_HOST_DEVICE inline TurnState50 make_turn_state50(
    InitialDeal50 deal,
    std::uint8_t first_player = 0) noexcept {
  collect_initial_floor_bonuses(deal, first_player);
  return TurnState50{deal.hand0,
                     deal.hand1,
                     deal.floor,
                     deal.stock,
                     deal.captured0,
                     deal.captured1,
                     deal.rng_state,
                     0,
                     0,
                     0,
                     0,
                     0,
                     0,
                     static_cast<std::uint8_t>(first_player & 1u),
                     Turn50Phase::kPlay,
                     core::kInvalidCard,
                     core::kInvalidCard};
}

CUGO_HOST_DEVICE inline CardMask active_hand50(const TurnState50& state) noexcept {
  return state.actor == 0 ? state.hand0 : state.hand1;
}

CUGO_HOST_DEVICE inline CardMask standard_floor50(const TurnState50& state) noexcept {
  return state.floor & core::kStandardDeckMask;
}

CUGO_HOST_DEVICE inline CardMask floor_matches50(const TurnState50& state,
                                                 CardId card) noexcept {
  return core::matching_month_cards(standard_floor50(state), card);
}

CUGO_HOST_DEVICE inline std::uint16_t card_month_bit50(CardId card) noexcept {
  return static_cast<std::uint16_t>(std::uint16_t{1} << core::card_month(card));
}

CUGO_HOST_DEVICE inline CardMask ppuk_bonus_mask_for_month(
    const TurnState50& state,
    std::uint16_t month_bit) noexcept {
  CardMask result = 0;
  if ((state.bonus2_ppuk_months & month_bit) != 0) {
    result |= core::card_bit(core::kBonusTwoPi);
  }
  if ((state.bonus3_ppuk_months & month_bit) != 0) {
    result |= core::card_bit(core::kBonusThreePi);
  }
  return result;
}

CUGO_HOST_DEVICE inline void clear_ppuk_bonus_association(
    TurnState50& state,
    std::uint16_t month_bit) noexcept {
  state.bonus2_ppuk_months &= static_cast<std::uint16_t>(~month_bit);
  state.bonus3_ppuk_months &= static_cast<std::uint16_t>(~month_bit);
}

CUGO_HOST_DEVICE inline void attach_pending_bonuses_to_ppuk(
    TurnState50& state,
    std::uint16_t month_bit) noexcept {
  if ((state.pending_bonus_mask & core::card_bit(core::kBonusTwoPi)) != 0) {
    state.bonus2_ppuk_months |= month_bit;
  }
  if ((state.pending_bonus_mask & core::card_bit(core::kBonusThreePi)) != 0) {
    state.bonus3_ppuk_months |= month_bit;
  }
}

CUGO_HOST_DEVICE inline Turn50BonusPlayResult play_bonus_for_turn50(
    TurnState50& state,
    CardId bonus) noexcept {
  if (state.phase != Turn50Phase::kPlay) {
    return Turn50BonusPlayResult{Turn50Status::kWrongPhase,
                                 core::kInvalidCard,
                                 0};
  }
  CardMask& hand = state.actor == 0 ? state.hand0 : state.hand1;
  CardMask& captured = state.actor == 0 ? state.captured0 : state.captured1;
  const BonusPlayResult result =
      play_hand_bonus(hand, state.stock, captured, state.rng_state, bonus);
  switch (result.status) {
    case BonusPlayStatus::kOk:
      return Turn50BonusPlayResult{Turn50Status::kOk,
                                   result.replacement,
                                   result.pi_steal_count};
    case BonusPlayStatus::kCardNotInHand:
      return Turn50BonusPlayResult{Turn50Status::kCardNotInHand,
                                   core::kInvalidCard,
                                   0};
    case BonusPlayStatus::kStockEmpty:
      return Turn50BonusPlayResult{Turn50Status::kStockEmpty,
                                   core::kInvalidCard,
                                   0};
    case BonusPlayStatus::kNotBonus:
    default:
      return Turn50BonusPlayResult{Turn50Status::kInvalidCard,
                                   core::kInvalidCard,
                                   0};
  }
}

CUGO_HOST_DEVICE inline Turn50Status begin_regular_play50(TurnState50& state,
                                                          CardId card) noexcept {
  if (state.phase != Turn50Phase::kPlay) {
    return Turn50Status::kWrongPhase;
  }
  if (!core::is_standard_card(card)) {
    return Turn50Status::kInvalidCard;
  }
  const CardMask bit = core::card_bit(card);
  CardMask& hand = state.actor == 0 ? state.hand0 : state.hand1;
  if ((hand & bit) == 0) {
    return Turn50Status::kCardNotInHand;
  }
  hand &= ~bit;
  state.pending_played = card;
  state.pending_drawn = core::kInvalidCard;
  state.pending_bonus_mask = 0;
  state.phase = Turn50Phase::kDraw;
  return Turn50Status::kOk;
}

CUGO_HOST_DEVICE inline Turn50Status draw_for_turn50(TurnState50& state) noexcept {
  if (state.phase != Turn50Phase::kDraw) {
    return Turn50Status::kWrongPhase;
  }

  CardMask next_stock = state.stock;
  std::uint64_t next_rng = state.rng_state;
  const BonusFlipResult flip = draw_stock_with_bonus_chain(next_stock, next_rng);
  if (!core::is_standard_card(flip.standard_card)) {
    return Turn50Status::kStockEmpty;
  }

  state.stock = next_stock;
  state.rng_state = next_rng;
  state.pending_drawn = flip.standard_card;
  state.pending_bonus_mask = flip.pending_bonus_mask;
  state.phase = Turn50Phase::kResolve;
  return Turn50Status::kOk;
}

CUGO_HOST_DEVICE inline void clear_ppuk_month50(TurnState50& state,
                                                std::uint16_t month_bit) noexcept {
  state.ppuk_months &= static_cast<std::uint16_t>(~month_bit);
  state.ppuk_owner1_months &= static_cast<std::uint16_t>(~month_bit);
}

CUGO_HOST_DEVICE inline void record_ppuk_capture50(
    TurnState50& state,
    CardId card,
    Resolve50Result& result) noexcept {
  const std::uint16_t month_bit = card_month_bit50(card);
  if ((state.ppuk_months & month_bit) == 0) {
    return;
  }
  const std::uint8_t owner =
      (state.ppuk_owner1_months & month_bit) != 0 ? 1u : 0u;
  if (owner == state.actor) {
    ++result.captured_own_ppuk;
  } else {
    ++result.captured_opponent_ppuk;
  }
  clear_ppuk_month50(state, month_bit);
}

CUGO_HOST_DEVICE inline void capture_cards50(TurnState50& state,
                                             CardMask cards,
                                             Resolve50Result& result) noexcept {
  if (state.actor == 0) {
    state.captured0 |= cards;
  } else {
    state.captured1 |= cards;
  }
  result.captured_cards |= cards;
}

CUGO_HOST_DEVICE inline Resolve50Status resolve_card_against_floor50(
    TurnState50& state,
    CardId card,
    CardId choice,
    Resolve50Result& result) noexcept {
  const CardMask bit = core::card_bit(card);
  const CardMask matches = floor_matches50(state, card);
  const int match_count = core::card_count(matches);

  if (match_count == 0) {
    state.floor |= bit;
    return Resolve50Status::kOk;
  }

  CardMask floor_capture = 0;
  if (match_count == 1) {
    floor_capture = matches;
  } else if (match_count == 2) {
    if (choice == core::kInvalidCard) {
      return Resolve50Status::kChoiceRequired;
    }
    if (!core::is_standard_card(choice) ||
        (matches & core::card_bit(choice)) == 0) {
      return Resolve50Status::kInvalidChoice;
    }
    floor_capture = core::card_bit(choice);
  } else {
    floor_capture = matches;
    const std::uint16_t month_bit = card_month_bit50(card);
    if ((state.ppuk_months & month_bit) != 0) {
      floor_capture |= ppuk_bonus_mask_for_month(state, month_bit);
      clear_ppuk_bonus_association(state, month_bit);
      record_ppuk_capture50(state, card, result);
    }
  }

  state.floor &= ~floor_capture;
  capture_cards50(state, floor_capture | bit, result);
  return Resolve50Status::kOk;
}

CUGO_HOST_DEVICE inline void finish_resolved_turn50(TurnState50& state) noexcept {
  state.pending_played = core::kInvalidCard;
  state.pending_drawn = core::kInvalidCard;
  state.pending_bonus_mask = 0;
  state.phase = Turn50Phase::kPlay;
  state.actor ^= 1u;
  ++state.turn_index;
}

CUGO_HOST_DEVICE inline Resolve50Result resolve_turn50(
    TurnState50& state,
    Resolve50Choices choices = kNoResolve50Choices) noexcept {
  Resolve50Result result{Resolve50Status::kWrongPhase,
                         kResolve50EventNone,
                         0,
                         0,
                         0};
  if (state.phase != Turn50Phase::kResolve) {
    return result;
  }

  TurnState50 next = state;
  const CardId played = next.pending_played;
  const CardId drawn = next.pending_drawn;
  const CardMask original_floor = next.floor;
  const CardMask played_matches = floor_matches50(next, played);
  const int played_match_count = core::card_count(played_matches);
  const bool same_month = core::card_month(played) == core::card_month(drawn);
  const bool final_stock_flip = next.stock == 0;

  if (same_month && played_match_count == 1) {
    if (final_stock_flip) {
      result.status = Resolve50Status::kUnsupportedLastCardSpecial;
      return result;
    }

    next.floor |= core::card_bit(played) | core::card_bit(drawn) |
                  next.pending_bonus_mask;
    const std::uint16_t month_bit = card_month_bit50(played);
    next.ppuk_months |= month_bit;
    if (next.actor == 0) {
      next.ppuk_owner1_months &= static_cast<std::uint16_t>(~month_bit);
    } else {
      next.ppuk_owner1_months |= month_bit;
    }
    attach_pending_bonuses_to_ppuk(next, month_bit);
    result.events |= kResolve50EventPpuk;
  } else if (same_month && played_match_count == 2) {
    next.floor &= ~played_matches;
    capture_cards50(next,
                    played_matches | core::card_bit(played) |
                        core::card_bit(drawn) | next.pending_bonus_mask,
                    result);
    result.events |= kResolve50EventTtadak;
  } else if (same_month && played_match_count == 0) {
    if (final_stock_flip) {
      result.status = Resolve50Status::kUnsupportedLastCardSpecial;
      return result;
    }

    capture_cards50(next,
                    core::card_bit(played) | core::card_bit(drawn) |
                        next.pending_bonus_mask,
                    result);
    result.events |= kResolve50EventJjok;
  } else {
    Resolve50Status status = resolve_card_against_floor50(
        next, played, choices.played_match, result);
    if (status != Resolve50Status::kOk) {
      return Resolve50Result{status, kResolve50EventNone, 0, 0, 0};
    }

    status = resolve_card_against_floor50(
        next, drawn, choices.drawn_match, result);
    if (status != Resolve50Status::kOk) {
      return Resolve50Result{status, kResolve50EventNone, 0, 0, 0};
    }

    if (next.pending_bonus_mask != 0) {
      capture_cards50(next, next.pending_bonus_mask, result);
    }
  }

  if (original_floor != 0 && next.floor == 0 && result.captured_cards != 0) {
    result.events |= kResolve50EventSweep;
  }

  finish_resolved_turn50(next);
  state = next;
  result.status = Resolve50Status::kOk;
  return result;
}

CUGO_HOST_DEVICE inline bool is_valid_turn_state50(const TurnState50& state) noexcept {
  if (state.actor > 1u ||
      static_cast<std::uint8_t>(state.phase) >
          static_cast<std::uint8_t>(Turn50Phase::kResolve) ||
      (state.ppuk_months & ~kAllMonthBits50) != 0 ||
      (state.ppuk_owner1_months & ~state.ppuk_months) != 0 ||
      (state.bonus2_ppuk_months & ~state.ppuk_months) != 0 ||
      (state.bonus3_ppuk_months & ~state.ppuk_months) != 0 ||
      core::card_count(state.bonus2_ppuk_months) > 1 ||
      core::card_count(state.bonus3_ppuk_months) > 1 ||
      (state.pending_bonus_mask & ~core::kBonusCardMask) != 0) {
    return false;
  }

  const CardMask expected_floor_bonuses =
      (state.bonus2_ppuk_months != 0
           ? core::card_bit(core::kBonusTwoPi)
           : CardMask{0}) |
      (state.bonus3_ppuk_months != 0
           ? core::card_bit(core::kBonusThreePi)
           : CardMask{0});
  if ((state.floor & core::kBonusCardMask) != expected_floor_bonuses) {
    return false;
  }

  for (std::uint8_t month = 0; month < core::kMonthCount; ++month) {
    const std::uint16_t month_bit =
        static_cast<std::uint16_t>(std::uint16_t{1} << month);
    if ((state.ppuk_months & month_bit) != 0 &&
        core::card_count(standard_floor50(state) & core::month_mask(month)) != 3) {
      return false;
    }
  }

  if (state.phase == Turn50Phase::kPlay) {
    if (state.pending_played != core::kInvalidCard ||
        state.pending_drawn != core::kInvalidCard ||
        state.pending_bonus_mask != 0) {
      return false;
    }
  } else if (state.phase == Turn50Phase::kDraw) {
    if (!core::is_standard_card(state.pending_played) ||
        state.pending_drawn != core::kInvalidCard ||
        state.pending_bonus_mask != 0) {
      return false;
    }
  } else {
    if (!core::is_standard_card(state.pending_played) ||
        !core::is_standard_card(state.pending_drawn)) {
      return false;
    }
  }

  CardMask seen = 0;
#define CUGO_ADD_DISJOINT50(mask_value)                            \
  do {                                                            \
    const CardMask current = (mask_value);                        \
    if ((current & ~core::kShinMatgoDeckMask) != 0) return false; \
    if ((seen & current) != 0) return false;                      \
    seen |= current;                                              \
  } while (false)

  CUGO_ADD_DISJOINT50(state.hand0);
  CUGO_ADD_DISJOINT50(state.hand1);
  CUGO_ADD_DISJOINT50(state.floor);
  CUGO_ADD_DISJOINT50(state.stock);
  CUGO_ADD_DISJOINT50(state.captured0);
  CUGO_ADD_DISJOINT50(state.captured1);
  CUGO_ADD_DISJOINT50(state.pending_bonus_mask);

#undef CUGO_ADD_DISJOINT50

  if (core::is_standard_card(state.pending_played)) {
    const CardMask bit = core::card_bit(state.pending_played);
    if ((seen & bit) != 0) return false;
    seen |= bit;
  }
  if (core::is_standard_card(state.pending_drawn)) {
    const CardMask bit = core::card_bit(state.pending_drawn);
    if ((seen & bit) != 0) return false;
    seen |= bit;
  }

  return seen == core::kShinMatgoDeckMask;
}

CUGO_HOST_DEVICE inline TurnState50 load_turn_state50(
    const TurnStateSoA50& states,
    std::size_t game) noexcept {
  return TurnState50{states.hand0[game],
                     states.hand1[game],
                     states.floor[game],
                     states.stock[game],
                     states.captured0[game],
                     states.captured1[game],
                     states.rng_state[game],
                     states.pending_bonus_mask[game],
                     states.ppuk_months[game],
                     states.ppuk_owner1_months[game],
                     states.bonus2_ppuk_months[game],
                     states.bonus3_ppuk_months[game],
                     states.turn_index[game],
                     states.actor[game],
                     static_cast<Turn50Phase>(states.phase[game]),
                     states.pending_played[game],
                     states.pending_drawn[game]};
}

CUGO_HOST_DEVICE inline void store_turn_state50(const TurnStateSoA50& states,
                                                std::size_t game,
                                                const TurnState50& state) noexcept {
  states.hand0[game] = state.hand0;
  states.hand1[game] = state.hand1;
  states.floor[game] = state.floor;
  states.stock[game] = state.stock;
  states.captured0[game] = state.captured0;
  states.captured1[game] = state.captured1;
  states.rng_state[game] = state.rng_state;
  states.pending_bonus_mask[game] = state.pending_bonus_mask;
  states.ppuk_months[game] = state.ppuk_months;
  states.ppuk_owner1_months[game] = state.ppuk_owner1_months;
  states.bonus2_ppuk_months[game] = state.bonus2_ppuk_months;
  states.bonus3_ppuk_months[game] = state.bonus3_ppuk_months;
  states.turn_index[game] = state.turn_index;
  states.actor[game] = state.actor;
  states.phase[game] = static_cast<std::uint8_t>(state.phase);
  states.pending_played[game] = state.pending_played;
  states.pending_drawn[game] = state.pending_drawn;
}

}  // namespace cugo::game

#undef CUGO_HOST_DEVICE
