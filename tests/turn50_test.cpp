#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/deal.h"
#include "cugo/game/turn50.h"

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

cugo::game::TurnState50 make_fixture(
    cugo::core::CardMask floor,
    cugo::core::CardId played,
    cugo::core::CardId drawn,
    cugo::core::CardMask pending_bonus = 0,
    std::uint8_t actor = 0,
    std::uint16_t ppuk_months = 0,
    std::uint16_t ppuk_owner1 = 0,
    std::uint16_t bonus2_assoc = 0,
    std::uint16_t bonus3_assoc = 0) {
  using namespace cugo::core;
  using namespace cugo::game;
  const CardMask pending = card_bit(played) | card_bit(drawn) | pending_bonus;
  assert((floor & pending) == 0);
  const CardMask remaining = kShinMatgoDeckMask & ~(floor | pending);
  TurnState50 s{};
  s.floor = floor;
  s.stock = remaining;
  s.rng_state = 0x123456789abcdef0ULL;
  s.pending_bonus_mask = pending_bonus;
  s.ppuk_months = ppuk_months;
  s.ppuk_owner1_months = ppuk_owner1;
  s.bonus2_ppuk_months = bonus2_assoc;
  s.bonus3_ppuk_months = bonus3_assoc;
  s.actor = actor;
  s.phase = Turn50Phase::kResolve;
  s.pending_played = played;
  s.pending_drawn = drawn;
  assert(is_valid_turn_state50(s));
  return s;
}

void test_initial_state_and_hand_bonus() {
  using namespace cugo::core;
  using namespace cugo::game;
  bool saw_floor_bonus = false;
  bool saw_hand_bonus = false;
  for (std::uint64_t i = 0; i < 4096; ++i) {
    const auto deal = deal_shin_matgo_50(derive_seed(0x50305f696e6974ULL, i));
    auto state = make_turn_state50(deal, static_cast<std::uint8_t>(i & 1u));
    assert(is_valid_turn_state50(state));
    if ((deal.floor & kBonusCardMask) != 0) {
      saw_floor_bonus = true;
      const CardMask expected = deal.floor & kBonusCardMask;
      const CardMask captured = state.actor == 0 ? state.captured0 : state.captured1;
      assert((captured & expected) == expected);
      assert((state.floor & kBonusCardMask) == 0);
    }

    CardMask& hand = state.actor == 0 ? state.hand0 : state.hand1;
    const CardMask hand_bonuses = hand & kBonusCardMask;
    if (hand_bonuses != 0) {
      saw_hand_bonus = true;
      const CardId bonus = first_card(hand_bonuses);
      const auto result = play_bonus_for_turn50(state, bonus);
      assert(result.status == Turn50Status::kOk);
      assert(result.pi_steal_count == 1);
      assert(is_physical_card(result.replacement));
      assert(state.phase == Turn50Phase::kPlay);
      assert(is_valid_turn_state50(state));
    }
  }
  assert(saw_floor_bonus);
  assert(saw_hand_bonus);
}

void test_pending_bonus_ppuk_association() {
  using namespace cugo::core;
  using namespace cugo::game;
  const CardMask both_bonuses = kBonusCardMask;
  auto state = make_fixture(card_bit(0), 1, 2, both_bonuses);
  const auto result = resolve_turn50(state);
  assert(result.status == Resolve50Status::kOk);
  assert((result.events & kResolve50EventPpuk) != 0);
  assert(result.captured_cards == 0);
  assert(state.floor == (card_bit(0) | card_bit(1) | card_bit(2) |
                         both_bonuses));
  assert((state.ppuk_months & 1u) != 0);
  assert((state.bonus2_ppuk_months & 1u) != 0);
  assert((state.bonus3_ppuk_months & 1u) != 0);
  assert(is_valid_turn_state50(state));
}

void test_capture_ppuk_captures_associated_bonus() {
  using namespace cugo::core;
  using namespace cugo::game;
  const std::uint16_t month0 = 1u;
  const CardMask floor = card_bit(0) | card_bit(1) | card_bit(2) |
                         card_bit(kBonusTwoPi) | card_bit(kBonusThreePi);
  auto state = make_fixture(floor, 3, 4, 0, 1, month0, 0,
                            month0, month0);
  const auto result = resolve_turn50(state);
  assert(result.status == Resolve50Status::kOk);
  const CardMask captured_month = month_mask(0) | kBonusCardMask;
  assert((result.captured_cards & captured_month) == captured_month);
  assert((state.captured1 & captured_month) == captured_month);
  assert(result.captured_opponent_ppuk == 1);
  assert(state.ppuk_months == 0);
  assert(state.bonus2_ppuk_months == 0);
  assert(state.bonus3_ppuk_months == 0);
  assert((state.floor & kBonusCardMask) == 0);
  assert(is_valid_turn_state50(state));
}

void test_non_ppuk_stock_bonus_is_captured() {
  using namespace cugo::core;
  using namespace cugo::game;
  auto state = make_fixture(card_bit(0) | card_bit(4), 1, 8,
                            card_bit(kBonusTwoPi));
  const auto result = resolve_turn50(state);
  assert(result.status == Resolve50Status::kOk);
  assert((result.captured_cards & card_bit(kBonusTwoPi)) != 0);
  assert((state.captured0 & card_bit(kBonusTwoPi)) != 0);
  assert((state.floor & kBonusCardMask) == 0);
  assert(is_valid_turn_state50(state));
}

void test_choice_is_transactional_with_pending_bonus() {
  using namespace cugo::core;
  using namespace cugo::game;
  auto state = make_fixture(card_bit(0) | card_bit(1) | card_bit(4), 2, 8,
                            card_bit(kBonusThreePi));
  const auto before = state;
  auto result = resolve_turn50(state);
  assert(result.status == Resolve50Status::kChoiceRequired);
  assert(result.captured_cards == 0);
  assert(same_state(state, before));
  result = resolve_turn50(state, Resolve50Choices{0, kInvalidCard});
  assert(result.status == Resolve50Status::kOk);
  assert((result.captured_cards & card_bit(kBonusThreePi)) != 0);
  assert(is_valid_turn_state50(state));
}

void test_random_first_turns() {
  using namespace cugo::core;
  using namespace cugo::game;
  int resolved = 0;
  int choices = 0;
  int hand_bonus = 0;
  int stock_bonus = 0;
  int ppuk_with_bonus = 0;
  for (std::uint64_t i = 0; i < 4096; ++i) {
    auto state = make_turn_state50(
        deal_shin_matgo_50(derive_seed(0x50305f7475726eULL, i)),
        static_cast<std::uint8_t>(i & 1u));
    assert(is_valid_turn_state50(state));

    CardMask bonuses = active_hand50(state) & kBonusCardMask;
    if (bonuses != 0) {
      ++hand_bonus;
      const auto br = play_bonus_for_turn50(state, first_card(bonuses));
      assert(br.status == Turn50Status::kOk);
      assert(is_valid_turn_state50(state));
    }

    const CardMask standards = active_hand50(state) & kStandardDeckMask;
    assert(standards != 0);
    assert(begin_regular_play50(state, first_card(standards)) == Turn50Status::kOk);
    assert(draw_for_turn50(state) == Turn50Status::kOk);
    if (state.pending_bonus_mask != 0) ++stock_bonus;
    const auto before = state;
    const auto rr = resolve_turn50(state);
    if (rr.status == Resolve50Status::kOk) {
      ++resolved;
      if ((rr.events & kResolve50EventPpuk) != 0 &&
          (state.floor & kBonusCardMask) != 0) {
        ++ppuk_with_bonus;
      }
      assert(is_valid_turn_state50(state));
    } else {
      assert(rr.status == Resolve50Status::kChoiceRequired);
      ++choices;
      assert(same_state(state, before));
    }
  }
  assert(resolved > 0);
  assert(choices > 0);
  assert(hand_bonus > 0);
  assert(stock_bonus > 0);
  assert(ppuk_with_bonus > 0);
}

}  // namespace

int main() {
  test_initial_state_and_hand_bonus();
  test_pending_bonus_ppuk_association();
  test_capture_ppuk_captures_associated_bonus();
  test_non_ppuk_stock_bonus_is_captured();
  test_choice_is_transactional_with_pending_bonus();
  test_random_first_turns();
  std::cout << "cugo_turn50_test: PASS\n";
  return 0;
}
