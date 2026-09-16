#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/game/hwatu.h"
#include "cugo/game/pi_transfer.h"

namespace {

bool same_state(const cugo::game::TurnState50& a,
                const cugo::game::TurnState50& b) {
  return a.hand0 == b.hand0 && a.hand1 == b.hand1 && a.floor == b.floor &&
         a.stock == b.stock && a.captured0 == b.captured0 &&
         a.captured1 == b.captured1 && a.rng_state == b.rng_state &&
         a.pending_bonus_mask == b.pending_bonus_mask &&
         a.ppuk_months == b.ppuk_months &&
         a.ppuk_owner1_months == b.ppuk_owner1_months &&
         a.bonus2_ppuk_months == b.bonus2_ppuk_months &&
         a.bonus3_ppuk_months == b.bonus3_ppuk_months &&
         a.turn_index == b.turn_index && a.actor == b.actor &&
         a.phase == b.phase && a.pending_played == b.pending_played &&
         a.pending_drawn == b.pending_drawn;
}

void test_steal_count_from_resolve_events() {
  using namespace cugo::game;
  Resolve50Result result{Resolve50Status::kOk,
                         kResolve50EventJjok,
                         0,
                         0,
                         0};
  assert(resolve_pi_steal_card_count(result) == 1);

  result.events = static_cast<std::uint8_t>(kResolve50EventTtadak |
                                            kResolve50EventSweep);
  assert(resolve_pi_steal_card_count(result) == 2);

  result.events = kResolve50EventSweep;
  result.captured_own_ppuk = 1;
  result.captured_opponent_ppuk = 1;
  assert(resolve_pi_steal_card_count(result) == 4);

  result.status = Resolve50Status::kChoiceRequired;
  assert(resolve_pi_steal_card_count(result) == 0);
}

void test_selection_boundary_and_transfer() {
  using namespace cugo::core;
  using namespace cugo::game;
  CardMask pi = kPlainPiMask;
  const CardId first = pop_first_card(pi);
  const CardId second = pop_first_card(pi);

  TurnState50 state{};
  state.captured1 = card_bit(first) | card_bit(second);
  const auto before = state;

  auto result = apply_pi_steal50(state, 0, 1);
  assert(result.status == PiTransferStatus::kSelectionRequired);
  assert(result.requested_cards == 1);
  assert(result.available_cards == 2);
  assert(same_state(state, before));

  result = apply_pi_steal50(
      state, 0, 1, PiTransferSelection{card_bit(first)});
  assert(result.status == PiTransferStatus::kOk);
  assert(result.transferred_cards == 1);
  assert(result.transferred_pi_units == 1);
  assert(result.transferred_mask == card_bit(first));
  assert((state.captured0 & card_bit(first)) != 0);
  assert((state.captured1 & card_bit(first)) == 0);
  assert((state.captured1 & card_bit(second)) != 0);
}

void test_invalid_selection_is_transactional() {
  using namespace cugo::core;
  using namespace cugo::game;
  CardMask pi = kPlainPiMask;
  const CardId first = pop_first_card(pi);
  const CardId second = pop_first_card(pi);
  const CardId non_pi = make_card(0, 0);

  TurnState50 state{};
  state.captured1 = card_bit(first) | card_bit(second) | card_bit(non_pi);
  const auto before = state;

  const auto result = apply_pi_steal50(
      state, 0, 1, PiTransferSelection{card_bit(non_pi)});
  assert(result.status == PiTransferStatus::kInvalidSelection);
  assert(same_state(state, before));
}

void test_take_all_when_victim_has_too_few_pi_cards() {
  using namespace cugo::core;
  using namespace cugo::game;
  const CardId double_pi = first_card(kFixedDoublePiMask);
  TurnState50 state{};
  state.captured1 = card_bit(double_pi);

  const auto result = apply_pi_steal50(state, 0, 2);
  assert(result.status == PiTransferStatus::kOk);
  assert(result.requested_cards == 2);
  assert(result.available_cards == 1);
  assert(result.transferred_cards == 1);
  assert(result.transferred_pi_units == 2);
  assert(result.transferred_mask == card_bit(double_pi));
  assert(state.captured1 == 0);
  assert(state.captured0 == card_bit(double_pi));
}

void test_bonus_and_gukjin_pi_values_follow_selected_card() {
  using namespace cugo::core;
  using namespace cugo::game;

  TurnState50 bonus_state{};
  bonus_state.captured1 = card_bit(kBonusThreePi);
  auto result = apply_pi_steal50(bonus_state, 0, 1);
  assert(result.status == PiTransferStatus::kOk);
  assert(result.transferred_pi_units == 3);

  TurnState50 gukjin_state{};
  gukjin_state.captured1 = card_bit(kGukjin);
  result = apply_pi_steal50(gukjin_state, 0, 1);
  assert(result.status == PiTransferStatus::kOk);
  assert(result.available_cards == 0);
  assert(gukjin_state.captured1 == card_bit(kGukjin));

  result = apply_pi_steal50(
      gukjin_state, 0, 1, kNoPiTransferSelection, ScoreOptions{true});
  assert(result.status == PiTransferStatus::kOk);
  assert(result.transferred_pi_units == 2);
  assert(gukjin_state.captured1 == 0);
  assert(gukjin_state.captured0 == card_bit(kGukjin));
}

void test_resolve_wrapper_is_transactional_on_pi_selection() {
  using namespace cugo::core;
  using namespace cugo::game;
  const CardId first = make_card(2, 2);
  const CardId second = make_card(3, 2);
  const CardId played = make_card(0, 0);
  const CardId drawn = make_card(0, 1);
  const CardId floor_card = make_card(1, 0);

  TurnState50 state{};
  state.actor = 0;
  state.phase = Turn50Phase::kResolve;
  state.floor = card_bit(floor_card);
  state.captured1 = card_bit(first) | card_bit(second);
  state.pending_played = played;
  state.pending_drawn = drawn;
  state.stock = kShinMatgoDeckMask &
                ~(state.floor | state.captured1 | card_bit(played) |
                  card_bit(drawn));
  assert(is_valid_turn_state50(state));
  const auto before = state;

  auto result = resolve_turn50_with_pi_transfer(state);
  assert(result.resolve.status == Resolve50Status::kOk);
  assert((result.resolve.events & kResolve50EventJjok) != 0);
  assert(result.pi_transfer.status == PiTransferStatus::kSelectionRequired);
  assert(same_state(state, before));

  result = resolve_turn50_with_pi_transfer(
      state,
      kNoResolve50Choices,
      PiTransferSelection{card_bit(second)});
  assert(result.resolve.status == Resolve50Status::kOk);
  assert(result.pi_transfer.status == PiTransferStatus::kOk);
  assert((state.captured0 & card_bit(second)) != 0);
  assert((state.captured1 & card_bit(second)) == 0);
}

void test_hand_bonus_wrapper_applies_one_pi_steal() {
  using namespace cugo::core;
  using namespace cugo::game;
  const CardId victim_pi = make_card(1, 2);
  const CardId replacement = make_card(0, 0);

  TurnState50 state{};
  state.actor = 0;
  state.phase = Turn50Phase::kPlay;
  state.hand0 = card_bit(kBonusTwoPi);
  state.stock = card_bit(replacement);
  state.captured1 = card_bit(victim_pi);
  state.rng_state = 0x123456789abcdef0ULL;

  const auto result = play_bonus_for_turn50_with_pi_transfer(
      state, kBonusTwoPi);
  assert(result.play.status == Turn50Status::kOk);
  assert(result.pi_transfer.status == PiTransferStatus::kOk);
  assert(result.pi_transfer.transferred_cards == 1);
  assert(result.pi_transfer.transferred_mask == card_bit(victim_pi));
  assert((state.captured0 & card_bit(victim_pi)) != 0);
  assert((state.captured1 & card_bit(victim_pi)) == 0);
}

}  // namespace

int main() {
  test_steal_count_from_resolve_events();
  test_selection_boundary_and_transfer();
  test_invalid_selection_is_transactional();
  test_take_all_when_victim_has_too_few_pi_cards();
  test_bonus_and_gukjin_pi_values_follow_selected_card();
  test_resolve_wrapper_is_transactional_on_pi_selection();
  test_hand_bonus_wrapper_applies_one_pi_steal();
  std::cout << "cugo_pi_transfer_test: PASS\n";
  return 0;
}
