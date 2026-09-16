#pragma once
#include <cstdint>
#include "cugo/game/game50.h"

#if defined(__CUDACC__)
#define CUGO_HD __host__ __device__
#else
#define CUGO_HD
#endif

namespace cugo::game {
using core::CardId;
using core::CardMask;

enum class SpecialKind50 : std::uint8_t { kShake = 0, kBomb = 1, kGrenade = 2, kBombCredit = 3 };
enum class SpecialStatus50 : std::uint8_t {
  kOk = 0, kWrongPhase, kInvalidCard, kInvalidMonth, kNotLegal, kStockEmpty,
  kResolveChoiceRequired, kInvalidResolveChoice, kPiSelectionRequired, kInvalidPiSelection
};
enum SpecialEvent50 : std::uint8_t {
  kSpecialNone = 0, kSpecialShake = 1u << 0, kSpecialBomb = 1u << 1,
  kSpecialGrenade = 1u << 2, kSpecialBombCredit = 1u << 3, kSpecialSweep = 1u << 4
};

struct SpecialGameState50 {
  GameState50 game;
  std::uint16_t shaken0, shaken1;
  std::uint8_t bombs0, bombs1;
  std::uint8_t credits0, credits1;
};
struct SpecialAction50 {
  SpecialKind50 kind;
  CardId card;
  std::uint8_t month;
  CardId drawn_match;
  PiTransferSelection pi_selection;
};
struct SpecialResult50 {
  SpecialStatus50 status;
  std::uint8_t events, actor, base_score, decision_opened, pi_steal_cards;
  CardId drawn;
  CardMask captured, stock_bonuses;
  PiTransferResult pi_transfer;
};

CUGO_HD inline SpecialGameState50 make_special_game_state50(InitialDeal50 d, std::uint8_t first = 0) noexcept {
  return SpecialGameState50{make_game_state50(d, first), 0, 0, 0, 0, 0, 0};
}
CUGO_HD inline std::uint16_t shaken50(const SpecialGameState50& s, std::uint8_t p) noexcept {
  return (p & 1u) == 0 ? s.shaken0 : s.shaken1;
}
CUGO_HD inline std::uint8_t bombs50(const SpecialGameState50& s, std::uint8_t p) noexcept {
  return (p & 1u) == 0 ? s.bombs0 : s.bombs1;
}
CUGO_HD inline std::uint8_t credits50(const SpecialGameState50& s, std::uint8_t p) noexcept {
  return (p & 1u) == 0 ? s.credits0 : s.credits1;
}
CUGO_HD inline CardMask month_cards50(CardMask cards, std::uint8_t m) noexcept {
  return m < core::kMonthCount ? cards & core::month_mask(m) : 0;
}
CUGO_HD inline bool special_ready50(const SpecialGameState50& s) noexcept {
  return !game50_is_finished(s.game) && !game50_has_pending_decision(s.game) &&
         s.game.turn.phase == Turn50Phase::kPlay;
}

CUGO_HD inline std::uint16_t legal_special_months50(const SpecialGameState50& s,
                                                     std::uint8_t hand_count,
                                                     std::uint8_t floor_count,
                                                     bool exclude_shaken) noexcept {
  if (!special_ready50(s)) return 0;
  const auto& t = s.game.turn;
  const CardMask h = active_hand50(t) & core::kStandardDeckMask;
  const CardMask f = standard_floor50(t);
  const std::uint16_t old = exclude_shaken ? shaken50(s, t.actor) : 0;
  std::uint16_t out = 0;
  for (std::uint8_t m = 0; m < core::kMonthCount; ++m) {
    const std::uint16_t bit = static_cast<std::uint16_t>(std::uint16_t{1} << m);
    if ((old & bit) == 0 && core::card_count(month_cards50(h, m)) == hand_count &&
        core::card_count(month_cards50(f, m)) == floor_count) out |= bit;
  }
  return out;
}
CUGO_HD inline std::uint16_t legal_shake_months50(const SpecialGameState50& s) noexcept {
  return legal_special_months50(s, 3, 0, true);
}
CUGO_HD inline std::uint16_t legal_bomb_months50(const SpecialGameState50& s) noexcept {
  return legal_special_months50(s, 3, 1, false);
}
CUGO_HD inline std::uint16_t legal_grenade_months50(const SpecialGameState50& s) noexcept {
  return legal_special_months50(s, 2, 2, false);
}
CUGO_HD inline std::uint8_t shake_count50(const SpecialGameState50& s, std::uint8_t p) noexcept {
  std::uint16_t x = shaken50(s, p); std::uint8_t n = 0;
  while (x) { x = static_cast<std::uint16_t>(x & (x - 1u)); ++n; }
  return n;
}
CUGO_HD inline std::uint64_t bomb_shake_multiplier50(const SpecialGameState50& s,
                                                      std::uint8_t p) noexcept {
  std::uint8_t n = static_cast<std::uint8_t>(bombs50(s, p) + shake_count50(s, p));
  std::uint64_t x = 1; while (n--) x *= 2u; return x;
}
CUGO_HD inline std::uint64_t go_bomb_shake_score50(const SpecialGameState50& s,
                                                    std::uint8_t p) noexcept {
  return go_adjusted_score_player50(s.game, p).points_after_go * bomb_shake_multiplier50(s, p);
}

CUGO_HD inline SpecialResult50 special_error50(SpecialStatus50 st) noexcept {
  return SpecialResult50{st, 0, kNoGame50Player, 0, 0, 0, core::kInvalidCard, 0, 0,
                         no_pi_transfer_result()};
}
CUGO_HD inline SpecialStatus50 pi_status50(PiTransferStatus s) noexcept {
  return s == PiTransferStatus::kSelectionRequired ? SpecialStatus50::kPiSelectionRequired :
         s == PiTransferStatus::kInvalidSelection ? SpecialStatus50::kInvalidPiSelection :
         SpecialStatus50::kOk;
}

struct DrawOnlySpecial50 {
  SpecialStatus50 status;
  std::uint8_t events;
  CardId drawn;
  CardMask bonuses;
  Resolve50Result resolve;
};
CUGO_HD inline DrawOnlySpecial50 draw_only_special50(TurnState50& t, CardId choice) noexcept {
  CardMask stock = t.stock; std::uint64_t rng = t.rng_state;
  const BonusFlipResult flip = draw_stock_with_bonus_chain(stock, rng);
  if (!core::is_standard_card(flip.standard_card))
    return DrawOnlySpecial50{SpecialStatus50::kStockEmpty, 0, core::kInvalidCard, 0,
      Resolve50Result{Resolve50Status::kWrongPhase, 0, 0, 0, 0}};
  t.stock = stock; t.rng_state = rng;
  const CardMask old_floor = t.floor;
  Resolve50Result rr{Resolve50Status::kOk, 0, 0, 0, 0};
  const Resolve50Status rs = resolve_card_against_floor50(t, flip.standard_card, choice, rr);
  if (rs != Resolve50Status::kOk) {
    const SpecialStatus50 st = rs == Resolve50Status::kChoiceRequired
      ? SpecialStatus50::kResolveChoiceRequired : SpecialStatus50::kInvalidResolveChoice;
    return DrawOnlySpecial50{st, 0, flip.standard_card, flip.pending_bonus_mask,
      Resolve50Result{rs, 0, 0, 0, 0}};
  }
  if (flip.pending_bonus_mask) capture_cards50(t, flip.pending_bonus_mask, rr);
  std::uint8_t ev = 0;
  if (old_floor && t.floor == 0 && rr.captured_cards) { rr.events |= kResolve50EventSweep; ev |= kSpecialSweep; }
  rr.status = Resolve50Status::kOk;
  return DrawOnlySpecial50{SpecialStatus50::kOk, ev, flip.standard_card, flip.pending_bonus_mask, rr};
}
CUGO_HD inline void finish_draw_only50(TurnState50& t) noexcept {
  t.pending_played = core::kInvalidCard; t.pending_drawn = core::kInvalidCard;
  t.pending_bonus_mask = 0; t.phase = Turn50Phase::kPlay; t.actor ^= 1u; ++t.turn_index;
}

CUGO_HD inline SpecialResult50 begin_shake50(SpecialGameState50& s, CardId card) noexcept {
  if (!special_ready50(s)) return special_error50(SpecialStatus50::kWrongPhase);
  if (!core::is_standard_card(card)) return special_error50(SpecialStatus50::kInvalidCard);
  const std::uint8_t m = core::card_month(card);
  const std::uint16_t bit = static_cast<std::uint16_t>(std::uint16_t{1} << m);
  if ((legal_shake_months50(s) & bit) == 0 ||
      (active_hand50(s.game.turn) & core::card_bit(card)) == 0)
    return special_error50(SpecialStatus50::kNotLegal);
  SpecialGameState50 n = s;
  if (n.game.turn.actor == 0) n.shaken0 |= bit; else n.shaken1 |= bit;
  if (begin_regular_play_game50(n.game, card) != Turn50Status::kOk)
    return special_error50(SpecialStatus50::kNotLegal);
  s = n;
  return SpecialResult50{SpecialStatus50::kOk, kSpecialShake, kNoGame50Player, 0, 0, 0,
                         core::kInvalidCard, 0, 0, no_pi_transfer_result()};
}

CUGO_HD inline SpecialResult50 capture_special50(SpecialGameState50& s,
                                                  std::uint8_t month,
                                                  bool grenade,
                                                  CardId choice,
                                                  PiTransferSelection sel) noexcept {
  if (!special_ready50(s)) return special_error50(SpecialStatus50::kWrongPhase);
  if (month >= core::kMonthCount) return special_error50(SpecialStatus50::kInvalidMonth);
  const std::uint16_t bit = static_cast<std::uint16_t>(std::uint16_t{1} << month);
  const std::uint16_t legal = grenade ? legal_grenade_months50(s) : legal_bomb_months50(s);
  if ((legal & bit) == 0) return special_error50(SpecialStatus50::kNotLegal);

  SpecialGameState50 n = s; TurnState50& t = n.game.turn; const std::uint8_t actor = t.actor;
  CardMask& hand = actor == 0 ? t.hand0 : t.hand1;
  const CardMask hc = month_cards50(hand, month), fc = month_cards50(standard_floor50(t), month);
  const CardMask immediate = hc | fc; hand &= ~hc; t.floor &= ~fc;
  if (actor == 0) { t.captured0 |= immediate; n.credits0 = static_cast<std::uint8_t>(n.credits0 + (grenade ? 1u : 2u)); if (!grenade) ++n.bombs0; }
  else { t.captured1 |= immediate; n.credits1 = static_cast<std::uint8_t>(n.credits1 + (grenade ? 1u : 2u)); if (!grenade) ++n.bombs1; }

  const DrawOnlySpecial50 d = draw_only_special50(t, choice);
  if (d.status != SpecialStatus50::kOk) return special_error50(d.status);
  const std::uint8_t steal = static_cast<std::uint8_t>(1u + resolve_pi_steal_card_count(d.resolve));
  finish_draw_only50(t);
  const PiTransferResult pt = apply_pi_steal50(t, actor, steal, sel);
  if (pt.status != PiTransferStatus::kOk) {
    SpecialResult50 r = special_error50(pi_status50(pt.status));
    r.pi_transfer = pt; r.pi_steal_cards = steal; r.drawn = d.drawn; r.stock_bonuses = d.bonuses; return r;
  }
  const std::uint8_t score = game50_base_score(n.game, actor);
  const bool opened = maybe_open_go_stop_decision50(n.game, actor);
  s = n;
  return SpecialResult50{SpecialStatus50::kOk,
    static_cast<std::uint8_t>((grenade ? kSpecialGrenade : kSpecialBomb) | d.events), actor,
    score, static_cast<std::uint8_t>(opened), steal, d.drawn,
    immediate | d.resolve.captured_cards, d.bonuses, pt};
}

CUGO_HD inline SpecialResult50 play_bomb50(SpecialGameState50& s, std::uint8_t m,
                                            CardId choice = core::kInvalidCard,
                                            PiTransferSelection sel = kNoPiTransferSelection) noexcept {
  return capture_special50(s, m, false, choice, sel);
}
CUGO_HD inline SpecialResult50 play_grenade50(SpecialGameState50& s, std::uint8_t m,
                                               CardId choice = core::kInvalidCard,
                                               PiTransferSelection sel = kNoPiTransferSelection) noexcept {
  return capture_special50(s, m, true, choice, sel);
}
CUGO_HD inline SpecialResult50 play_bomb_credit50(SpecialGameState50& s,
                                                   CardId choice = core::kInvalidCard,
                                                   PiTransferSelection sel = kNoPiTransferSelection) noexcept {
  if (!special_ready50(s)) return special_error50(SpecialStatus50::kWrongPhase);
  const std::uint8_t actor = s.game.turn.actor;
  if (!credits50(s, actor)) return special_error50(SpecialStatus50::kNotLegal);
  SpecialGameState50 n = s; if (actor == 0) --n.credits0; else --n.credits1;
  const DrawOnlySpecial50 d = draw_only_special50(n.game.turn, choice);
  if (d.status != SpecialStatus50::kOk) return special_error50(d.status);
  const std::uint8_t steal = resolve_pi_steal_card_count(d.resolve);
  finish_draw_only50(n.game.turn);
  const PiTransferResult pt = apply_pi_steal50(n.game.turn, actor, steal, sel);
  if (pt.status != PiTransferStatus::kOk) {
    SpecialResult50 r = special_error50(pi_status50(pt.status));
    r.pi_transfer = pt; r.pi_steal_cards = steal; r.drawn = d.drawn; r.stock_bonuses = d.bonuses; return r;
  }
  const std::uint8_t score = game50_base_score(n.game, actor);
  const bool opened = maybe_open_go_stop_decision50(n.game, actor);
  s = n;
  return SpecialResult50{SpecialStatus50::kOk,
    static_cast<std::uint8_t>(kSpecialBombCredit | d.events), actor, score,
    static_cast<std::uint8_t>(opened), steal, d.drawn, d.resolve.captured_cards, d.bonuses, pt};
}

CUGO_HD inline SpecialResult50 apply_special_action50(SpecialGameState50& s,
                                                       SpecialAction50 a) noexcept {
  switch (a.kind) {
    case SpecialKind50::kShake: return begin_shake50(s, a.card);
    case SpecialKind50::kBomb: return play_bomb50(s, a.month, a.drawn_match, a.pi_selection);
    case SpecialKind50::kGrenade: return play_grenade50(s, a.month, a.drawn_match, a.pi_selection);
    case SpecialKind50::kBombCredit: return play_bomb_credit50(s, a.drawn_match, a.pi_selection);
    default: return special_error50(SpecialStatus50::kNotLegal);
  }
}
CUGO_HD inline bool is_valid_special_game_state50(const SpecialGameState50& s) noexcept {
  return is_valid_game_state50(s.game) &&
         (s.shaken0 & ~kAllMonthBits50) == 0 && (s.shaken1 & ~kAllMonthBits50) == 0;
}

}  // namespace cugo::game
#undef CUGO_HD
