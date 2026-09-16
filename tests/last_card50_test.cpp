#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/game/pi_transfer.h"
#include "cugo/game/turn50.h"

namespace {

cugo::game::TurnState50 make_resolve_fixture(
    cugo::core::CardMask floor,
    cugo::core::CardId played,
    cugo::core::CardId drawn,
    cugo::core::CardMask pending_bonus,
    bool final_stock_flip) {
  using namespace cugo::core;
  using namespace cugo::game;

  const CardMask pending = card_bit(played) | card_bit(drawn) | pending_bonus;
  assert((floor & pending) == 0);

  CardMask remaining = kShinMatgoDeckMask & ~(floor | pending);
  CardMask stock = 0;
  if (!final_stock_flip) {
    const CardId stock_card = first_card(remaining & kStandardDeckMask);
    assert(is_standard_card(stock_card));
    stock = card_bit(stock_card);
    remaining &= ~stock;
  }

  TurnState50 state{};
  state.floor = floor;
  state.stock = stock;
  state.captured1 = remaining;
  state.rng_state = 0x6c6173745f636172ULL;
  state.pending_bonus_mask = pending_bonus;
  state.actor = 0;
  state.phase = Turn50Phase::kResolve;
  state.pending_played = played;
  state.pending_drawn = drawn;
  assert(is_valid_turn_state50(state));
  return state;
}

void test_last_jjok_becomes_ordinary_capture() {
  using namespace cugo::core;
  using namespace cugo::game;

  auto state = make_resolve_fixture(card_bit(4), 0, 1,
                                    card_bit(kBonusTwoPi), true);
  const auto result = resolve_turn50(state);

  const CardMask expected = card_bit(0) | card_bit(1) |
                            card_bit(kBonusTwoPi);
  assert(result.status == Resolve50Status::kOk);
  assert((result.events & kResolve50EventJjok) == 0);
  assert((result.events & kResolve50EventPpuk) == 0);
  assert(result.captured_cards == expected);
  assert(resolve_pi_steal_card_count(result) == 0);
  assert(state.floor == card_bit(4));
  assert((state.captured0 & expected) == expected);
  assert(state.ppuk_months == 0);
  assert(state.phase == Turn50Phase::kPlay);
  assert(state.actor == 1);
  assert(is_valid_turn_state50(state));
}

void test_last_ppuk_candidate_resolves_sequentially() {
  using namespace cugo::core;
  using namespace cugo::game;

  auto state = make_resolve_fixture(card_bit(0) | card_bit(4), 1, 2,
                                    card_bit(kBonusThreePi), true);
  const auto result = resolve_turn50(state);

  const CardMask expected_capture = card_bit(0) | card_bit(1) |
                                    card_bit(kBonusThreePi);
  const CardMask expected_floor = card_bit(2) | card_bit(4);
  assert(result.status == Resolve50Status::kOk);
  assert((result.events & kResolve50EventPpuk) == 0);
  assert((result.events & kResolve50EventJjok) == 0);
  assert(result.captured_cards == expected_capture);
  assert(resolve_pi_steal_card_count(result) == 0);
  assert(state.floor == expected_floor);
  assert((state.captured0 & expected_capture) == expected_capture);
  assert(state.ppuk_months == 0);
  assert(state.bonus2_ppuk_months == 0);
  assert(state.bonus3_ppuk_months == 0);
  assert(is_valid_turn_state50(state));
}

void test_non_final_ppuk_and_jjok_stay_special() {
  using namespace cugo::core;
  using namespace cugo::game;

  auto ppuk = make_resolve_fixture(card_bit(0) | card_bit(4), 1, 2, 0, false);
  const auto ppuk_result = resolve_turn50(ppuk);
  assert(ppuk_result.status == Resolve50Status::kOk);
  assert((ppuk_result.events & kResolve50EventPpuk) != 0);
  assert(ppuk_result.captured_cards == 0);
  assert((ppuk.ppuk_months & 1u) != 0);
  assert((ppuk.floor & (card_bit(0) | card_bit(1) | card_bit(2))) ==
         (card_bit(0) | card_bit(1) | card_bit(2)));
  assert(is_valid_turn_state50(ppuk));

  auto jjok = make_resolve_fixture(card_bit(4), 0, 1, 0, false);
  const auto jjok_result = resolve_turn50(jjok);
  assert(jjok_result.status == Resolve50Status::kOk);
  assert((jjok_result.events & kResolve50EventJjok) != 0);
  assert(jjok_result.captured_cards == (card_bit(0) | card_bit(1)));
  assert(resolve_pi_steal_card_count(jjok_result) == 1);
  assert(is_valid_turn_state50(jjok));
}

}  // namespace

int main() {
  test_last_jjok_becomes_ordinary_capture();
  test_last_ppuk_candidate_resolves_sequentially();
  test_non_final_ppuk_and_jjok_stay_special();
  std::cout << "cugo_last_card50_test: PASS\n";
  return 0;
}
