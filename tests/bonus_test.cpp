#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/bonus.h"
#include "cugo/game/deal.h"
#include "cugo/game/hwatu.h"

namespace {

void test_physical_deck() {
  using namespace cugo::core;
  static_assert(kStandardCardCount == 48);
  static_assert(kShinMatgoCardCount == 50);
  static_assert(kBonusTwoPi == 48);
  static_assert(kBonusThreePi == 49);
  static_assert(kStandardDeckMask == 0x0000ffffffffffffULL);
  static_assert(kShinMatgoDeckMask == 0x0003ffffffffffffULL);
  assert(is_standard_card(47));
  assert(!is_standard_card(kBonusTwoPi));
  assert(is_bonus_card(kBonusTwoPi));
  assert(is_bonus_card(kBonusThreePi));
  assert(is_physical_card(kBonusThreePi));
  assert(card_count(kBonusCardMask) == 2);
}

void test_raw_deal_and_initial_floor_bonus() {
  using namespace cugo::core;
  using namespace cugo::game;
  constexpr std::uint64_t master = 0x626f6e75735f3530ULL;
  int floor_bonus_deals = 0;
  for (std::uint64_t i = 0; i < 4096; ++i) {
    auto deal = deal_shin_matgo_50(derive_seed(master, i));
    assert(is_valid_initial_deal(deal));
    assert(card_count(deal.hand0) == 10);
    assert(card_count(deal.hand1) == 10);
    assert(card_count(deal.floor) == 8);
    assert(card_count(deal.stock) == 22);
    const CardMask floor_bonus = deal.floor & kBonusCardMask;
    if (floor_bonus != 0) ++floor_bonus_deals;
    const std::uint8_t first = static_cast<std::uint8_t>(i & 1u);
    const CardMask collected = collect_initial_floor_bonuses(deal, first);
    assert(collected == floor_bonus);
    assert((deal.floor & kBonusCardMask) == 0);
    assert((first == 0 ? deal.captured0 : deal.captured1) == floor_bonus);
    assert((first == 0 ? deal.captured1 : deal.captured0) == 0);
    assert(is_valid_initial_deal(deal));
  }
  assert(floor_bonus_deals > 0);
}

void test_hand_bonus_play() {
  using namespace cugo::core;
  using namespace cugo::game;
  CardMask hand = card_bit(kBonusTwoPi) | card_bit(0) | card_bit(4);
  CardMask stock = card_bit(8) | card_bit(9) | card_bit(kBonusThreePi);
  CardMask captured = 0;
  std::uint64_t rng = 0x123456789abcdef0ULL;
  const CardMask before_all = hand | stock | captured;
  const int before_hand = card_count(hand);
  const int before_stock = card_count(stock);
  const auto result = play_hand_bonus(hand, stock, captured, rng, kBonusTwoPi);
  assert(result.status == BonusPlayStatus::kOk);
  assert(is_physical_card(result.replacement));
  assert(result.pi_steal_count == 1);
  assert((captured & card_bit(kBonusTwoPi)) != 0);
  assert((hand & card_bit(kBonusTwoPi)) == 0);
  assert((hand & card_bit(result.replacement)) != 0);
  assert(card_count(hand) == before_hand);
  assert(card_count(stock) == before_stock - 1);
  assert((hand | stock | captured) == before_all);

  CardMask absent_hand = hand;
  CardMask empty_stock = 0;
  CardMask unchanged_captured = captured;
  const auto absent = play_hand_bonus(absent_hand, empty_stock, unchanged_captured,
                                      rng, kBonusThreePi);
  assert(absent.status == BonusPlayStatus::kCardNotInHand);

  CardMask bonus_hand = card_bit(kBonusThreePi);
  CardMask no_stock = 0;
  CardMask no_capture = 0;
  const std::uint64_t rng_before = rng;
  const auto empty = play_hand_bonus(bonus_hand, no_stock, no_capture,
                                     rng, kBonusThreePi);
  assert(empty.status == BonusPlayStatus::kStockEmpty);
  assert(bonus_hand == card_bit(kBonusThreePi));
  assert(no_stock == 0 && no_capture == 0 && rng == rng_before);
}

void test_stock_bonus_chain() {
  using namespace cugo::core;
  using namespace cugo::game;
  CardMask stock = card_bit(kBonusTwoPi) | card_bit(kBonusThreePi) |
                   card_bit(make_card(3, 2));
  std::uint64_t rng = 1;
  const auto result = draw_stock_with_bonus_chain(stock, rng);
  assert(result.pending_bonus_mask == kBonusCardMask);
  assert(result.standard_card == make_card(3, 2));
  assert(stock == 0);
}

void test_bonus_scoring() {
  using namespace cugo::core;
  using namespace cugo::game;
  const CardMask bonuses = kBonusCardMask;
  assert(pi_card_mask(bonuses) == bonuses);
  assert(pi_units(card_bit(kBonusTwoPi)) == 2);
  assert(pi_units(card_bit(kBonusThreePi)) == 3);
  assert(pi_units(bonuses) == 5);
  assert(score_captured(bonuses).pi_units == 5);

  const auto full = score_captured(kShinMatgoDeckMask);
  const auto converted = score_captured(kShinMatgoDeckMask, ScoreOptions{true});
  assert(full.pi_units == 31);
  assert(converted.pi_units == 33);
}

}  // namespace

int main() {
  test_physical_deck();
  test_raw_deal_and_initial_floor_bonus();
  test_hand_bonus_play();
  test_stock_bonus_chain();
  test_bonus_scoring();
  std::cout << "cugo_bonus_test: PASS\n";
  return 0;
}
