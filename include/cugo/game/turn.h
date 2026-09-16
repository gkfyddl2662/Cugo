#pragma once

#include <cstddef>
#include <cstdint>

#include "cugo/core/card.h"
#include "cugo/game/deal.h"
#include "cugo/game/state.h"

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::game {

using core::CardId;
using core::CardMask;

inline constexpr std::uint16_t kAllMonthBits =
    (std::uint16_t{1} << core::kMonthCount) - 1;

enum class TurnPhase : std::uint8_t {
  kPlay = 0,
  kDraw = 1,
  kResolve = 2,
};

enum class TurnStatus : std::uint8_t {
  kOk = 0,
  kWrongPhase = 1,
  kInvalidCard = 2,
  kCardNotInHand = 3,
  kStockEmpty = 4,
};

enum class ResolveStatus : std::uint8_t {
  kOk = 0,
  kWrongPhase = 1,
  kChoiceRequired = 2,
  kInvalidChoice = 3,
  kUnsupportedLastCardSpecial = 4,
};

enum ResolveEvent : std::uint8_t {
  kResolveEventNone = 0,
  kResolveEventPpuk = 1u << 0,
  kResolveEventJjok = 1u << 1,
  kResolveEventTtadak = 1u << 2,
  kResolveEventSweep = 1u << 3,
};

struct ResolveChoices {
  CardId played_match;
  CardId drawn_match;
};

inline constexpr ResolveChoices kNoResolveChoices{core::kInvalidCard,
                                                   core::kInvalidCard};

struct ResolveResult {
  ResolveStatus status;
  std::uint8_t events;
  std::uint8_t captured_own_ppuk;
  std::uint8_t captured_opponent_ppuk;
  CardMask captured_cards;
};

struct TurnState48 {
  CardMask hand0;
  CardMask hand1;
  CardMask floor;
  CardMask stock;
  CardMask captured0;
  CardMask captured1;
  std::uint64_t rng_state;
  std::uint16_t ppuk_months;
  std::uint16_t ppuk_owner1_months;
  std::uint16_t turn_index;
  std::uint8_t actor;
  TurnPhase phase;
  CardId pending_played;
  CardId pending_drawn;
};

struct TurnStateSoA48 {
  CardMask* hand0;
  CardMask* hand1;
  CardMask* floor;
  CardMask* stock;
  CardMask* captured0;
  CardMask* captured1;
  std::uint64_t* rng_state;
  std::uint16_t* ppuk_months;
  std::uint16_t* ppuk_owner1_months;
  std::uint16_t* turn_index;
  std::uint8_t* actor;
  std::uint8_t* phase;
  CardId* pending_played;
  CardId* pending_drawn;
};

CUGO_HOST_DEVICE inline TurnState48 make_turn_state(
    const InitialDeal48& deal,
    std::uint8_t first_player = 0) noexcept {
  return TurnState48{deal.hand0,
                     deal.hand1,
                     deal.floor,
                     deal.stock,
                     0,
                     0,
                     deal.rng_state,
                     0,
                     0,
                     0,
                     static_cast<std::uint8_t>(first_player & 1u),
                     TurnPhase::kPlay,
                     core::kInvalidCard,
                     core::kInvalidCard};
}

CUGO_HOST_DEVICE inline CardMask active_hand(const TurnState48& state) noexcept {
  return state.actor == 0 ? state.hand0 : state.hand1;
}

CUGO_HOST_DEVICE inline CardMask floor_matches(const TurnState48& state,
                                               CardId card) noexcept {
  return core::matching_month_cards(state.floor, card);
}

CUGO_HOST_DEVICE inline TurnStatus begin_regular_play(TurnState48& state,
                                                      CardId card) noexcept {
  if (state.phase != TurnPhase::kPlay) {
    return TurnStatus::kWrongPhase;
  }
  if (!core::is_valid_card(card)) {
    return TurnStatus::kInvalidCard;
  }

  const CardMask bit = core::card_bit(card);
  CardMask& hand = state.actor == 0 ? state.hand0 : state.hand1;
  if ((hand & bit) == 0) {
    return TurnStatus::kCardNotInHand;
  }

  hand &= ~bit;
  state.pending_played = card;
  state.pending_drawn = core::kInvalidCard;
  state.phase = TurnPhase::kDraw;
  return TurnStatus::kOk;
}

CUGO_HOST_DEVICE inline TurnStatus draw_for_turn(TurnState48& state) noexcept {
  if (state.phase != TurnPhase::kDraw) {
    return TurnStatus::kWrongPhase;
  }

  const CardId card = draw_stock_card(state.stock, state.rng_state);
  if (card == core::kInvalidCard) {
    return TurnStatus::kStockEmpty;
  }

  state.pending_drawn = card;
  state.phase = TurnPhase::kResolve;
  return TurnStatus::kOk;
}

CUGO_HOST_DEVICE inline std::uint16_t card_month_bit(CardId card) noexcept {
  return static_cast<std::uint16_t>(std::uint16_t{1} << core::card_month(card));
}

CUGO_HOST_DEVICE inline void clear_ppuk_month(TurnState48& state,
                                              std::uint16_t month_bit) noexcept {
  state.ppuk_months &= static_cast<std::uint16_t>(~month_bit);
  state.ppuk_owner1_months &= static_cast<std::uint16_t>(~month_bit);
}

CUGO_HOST_DEVICE inline void record_ppuk_capture(TurnState48& state,
                                                 CardId card,
                                                 ResolveResult& result) noexcept {
  const std::uint16_t month_bit = card_month_bit(card);
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
  clear_ppuk_month(state, month_bit);
}

CUGO_HOST_DEVICE inline void capture_cards(TurnState48& state,
                                           CardMask cards,
                                           ResolveResult& result) noexcept {
  if (state.actor == 0) {
    state.captured0 |= cards;
  } else {
    state.captured1 |= cards;
  }
  result.captured_cards |= cards;
}

CUGO_HOST_DEVICE inline ResolveStatus resolve_card_against_floor(
    TurnState48& state,
    CardId card,
    CardId choice,
    ResolveResult& result) noexcept {
  const CardMask bit = core::card_bit(card);
  const CardMask matches = core::matching_month_cards(state.floor, card);
  const int match_count = core::card_count(matches);

  if (match_count == 0) {
    state.floor |= bit;
    return ResolveStatus::kOk;
  }

  CardMask floor_capture = 0;
  if (match_count == 1) {
    floor_capture = matches;
  } else if (match_count == 2) {
    if (choice == core::kInvalidCard) {
      return ResolveStatus::kChoiceRequired;
    }
    if (!core::is_valid_card(choice) ||
        (matches & core::card_bit(choice)) == 0) {
      return ResolveStatus::kInvalidChoice;
    }
    floor_capture = core::card_bit(choice);
  } else {
    // Three cards of one month on the floor are captured together with the
    // fourth card. If this was a ppuk stack, keep its ownership side effect.
    floor_capture = matches;
    record_ppuk_capture(state, card, result);
  }

  state.floor &= ~floor_capture;
  capture_cards(state, floor_capture | bit, result);
  return ResolveStatus::kOk;
}

CUGO_HOST_DEVICE inline void finish_resolved_turn(TurnState48& state) noexcept {
  state.pending_played = core::kInvalidCard;
  state.pending_drawn = core::kInvalidCard;
  state.phase = TurnPhase::kPlay;
  state.actor ^= 1u;
  ++state.turn_index;
}

CUGO_HOST_DEVICE inline ResolveResult resolve_turn(
    TurnState48& state,
    ResolveChoices choices = kNoResolveChoices) noexcept {
  ResolveResult result{ResolveStatus::kWrongPhase,
                       kResolveEventNone,
                       0,
                       0,
                       0};
  if (state.phase != TurnPhase::kResolve) {
    return result;
  }

  TurnState48 next = state;
  const CardId played = next.pending_played;
  const CardId drawn = next.pending_drawn;
  const CardMask original_floor = next.floor;
  const CardMask played_matches =
      core::matching_month_cards(original_floor, played);
  const int played_match_count = core::card_count(played_matches);
  const bool same_month = core::card_month(played) == core::card_month(drawn);
  const bool final_stock_flip = next.stock == 0;

  if (same_month && played_match_count == 1) {
    if (final_stock_flip) {
      result.status = ResolveStatus::kUnsupportedLastCardSpecial;
      return result;
    }

    next.floor |= core::card_bit(played) | core::card_bit(drawn);
    const std::uint16_t month_bit = card_month_bit(played);
    next.ppuk_months |= month_bit;
    if (next.actor == 0) {
      next.ppuk_owner1_months &= static_cast<std::uint16_t>(~month_bit);
    } else {
      next.ppuk_owner1_months |= month_bit;
    }
    result.events |= kResolveEventPpuk;
  } else if (same_month && played_match_count == 2) {
    next.floor &= ~played_matches;
    capture_cards(next,
                  played_matches | core::card_bit(played) |
                      core::card_bit(drawn),
                  result);
    result.events |= kResolveEventTtadak;
  } else if (same_month && played_match_count == 0) {
    if (final_stock_flip) {
      result.status = ResolveStatus::kUnsupportedLastCardSpecial;
      return result;
    }

    capture_cards(next,
                  core::card_bit(played) | core::card_bit(drawn),
                  result);
    result.events |= kResolveEventJjok;
  } else {
    ResolveStatus status = resolve_card_against_floor(
        next, played, choices.played_match, result);
    if (status != ResolveStatus::kOk) {
      return ResolveResult{status, kResolveEventNone, 0, 0, 0};
    }

    status = resolve_card_against_floor(
        next, drawn, choices.drawn_match, result);
    if (status != ResolveStatus::kOk) {
      return ResolveResult{status, kResolveEventNone, 0, 0, 0};
    }
  }

  if (original_floor != 0 && next.floor == 0 && result.captured_cards != 0) {
    result.events |= kResolveEventSweep;
  }

  finish_resolved_turn(next);
  state = next;
  result.status = ResolveStatus::kOk;
  return result;
}

CUGO_HOST_DEVICE inline bool is_valid_turn_state(const TurnState48& state) noexcept {
  if (state.actor > 1u ||
      static_cast<std::uint8_t>(state.phase) >
          static_cast<std::uint8_t>(TurnPhase::kResolve) ||
      (state.ppuk_months & ~kAllMonthBits) != 0 ||
      (state.ppuk_owner1_months & ~state.ppuk_months) != 0) {
    return false;
  }

  for (std::uint8_t month = 0; month < core::kMonthCount; ++month) {
    const std::uint16_t month_bit =
        static_cast<std::uint16_t>(std::uint16_t{1} << month);
    if ((state.ppuk_months & month_bit) != 0 &&
        core::card_count(state.floor & core::month_mask(month)) != 3) {
      return false;
    }
  }

  if (state.phase == TurnPhase::kPlay) {
    if (state.pending_played != core::kInvalidCard ||
        state.pending_drawn != core::kInvalidCard) {
      return false;
    }
  } else if (state.phase == TurnPhase::kDraw) {
    if (!core::is_valid_card(state.pending_played) ||
        state.pending_drawn != core::kInvalidCard) {
      return false;
    }
  } else {
    if (!core::is_valid_card(state.pending_played) ||
        !core::is_valid_card(state.pending_drawn)) {
      return false;
    }
  }

  CardMask seen = 0;
#define CUGO_ADD_DISJOINT(mask_value)        \
  do {                                       \
    const CardMask current = (mask_value);   \
    if ((seen & current) != 0) return false; \
    seen |= current;                         \
  } while (false)

  CUGO_ADD_DISJOINT(state.hand0);
  CUGO_ADD_DISJOINT(state.hand1);
  CUGO_ADD_DISJOINT(state.floor);
  CUGO_ADD_DISJOINT(state.stock);
  CUGO_ADD_DISJOINT(state.captured0);
  CUGO_ADD_DISJOINT(state.captured1);

#undef CUGO_ADD_DISJOINT

  if (core::is_valid_card(state.pending_played)) {
    const CardMask bit = core::card_bit(state.pending_played);
    if ((seen & bit) != 0) {
      return false;
    }
    seen |= bit;
  }
  if (core::is_valid_card(state.pending_drawn)) {
    const CardMask bit = core::card_bit(state.pending_drawn);
    if ((seen & bit) != 0) {
      return false;
    }
    seen |= bit;
  }

  return seen == core::kFullDeckMask;
}

CUGO_HOST_DEVICE inline TurnState48 load_turn_state(
    const TurnStateSoA48& states,
    std::size_t game) noexcept {
  return TurnState48{states.hand0[game],
                     states.hand1[game],
                     states.floor[game],
                     states.stock[game],
                     states.captured0[game],
                     states.captured1[game],
                     states.rng_state[game],
                     states.ppuk_months[game],
                     states.ppuk_owner1_months[game],
                     states.turn_index[game],
                     states.actor[game],
                     static_cast<TurnPhase>(states.phase[game]),
                     states.pending_played[game],
                     states.pending_drawn[game]};
}

CUGO_HOST_DEVICE inline void store_turn_state(const TurnStateSoA48& states,
                                              std::size_t game,
                                              const TurnState48& state) noexcept {
  states.hand0[game] = state.hand0;
  states.hand1[game] = state.hand1;
  states.floor[game] = state.floor;
  states.stock[game] = state.stock;
  states.captured0[game] = state.captured0;
  states.captured1[game] = state.captured1;
  states.rng_state[game] = state.rng_state;
  states.ppuk_months[game] = state.ppuk_months;
  states.ppuk_owner1_months[game] = state.ppuk_owner1_months;
  states.turn_index[game] = state.turn_index;
  states.actor[game] = state.actor;
  states.phase[game] = static_cast<std::uint8_t>(state.phase);
  states.pending_played[game] = state.pending_played;
  states.pending_drawn[game] = state.pending_drawn;
}

}  // namespace cugo::game

#undef CUGO_HOST_DEVICE
