#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/deal.h"

namespace {

void test_card_layout() {
  using namespace cugo::core;
  static_assert(kCardCount == 48);
  static_assert(kFullDeckMask == 0x0000ffffffffffffULL);

  CardMask seen = 0;
  for (int month = 0; month < kMonthCount; ++month) {
    for (int slot = 0; slot < kCardsPerMonth; ++slot) {
      const CardId card = make_card(static_cast<std::uint8_t>(month), static_cast<std::uint8_t>(slot));
      assert(is_valid_card(card));
      assert(card_month(card) == month);
      assert(card_slot(card) == slot);
      seen |= card_bit(card);
    }
  }
  assert(seen == kFullDeckMask);
  assert(card_count(seen) == kCardCount);
}

void test_pop_first_card() {
  using namespace cugo::core;
  CardMask mask = card_bit(3) | card_bit(17) | card_bit(47);
  assert(pop_first_card(mask) == 3);
  assert(pop_first_card(mask) == 17);
  assert(pop_first_card(mask) == 47);
  assert(pop_first_card(mask) == kInvalidCard);
  assert(mask == 0);
}

void test_select_card_by_rank() {
  using namespace cugo::core;
  const CardMask mask = card_bit(0) | card_bit(7) | card_bit(31) | card_bit(32) | card_bit(47);
  assert(select_card_by_rank(mask, 0) == 0);
  assert(select_card_by_rank(mask, 1) == 7);
  assert(select_card_by_rank(mask, 2) == 31);
  assert(select_card_by_rank(mask, 3) == 32);
  assert(select_card_by_rank(mask, 4) == 47);
  assert(select_card_by_rank(mask, 5) == kInvalidCard);

  for (std::uint32_t rank = 0; rank < kCardCount; ++rank) {
    assert(select_card_by_rank(kFullDeckMask, rank) == rank);
  }
}

void test_rng_determinism() {
  using cugo::core::SplitMix64;
  SplitMix64 a{0x123456789abcdef0ULL};
  SplitMix64 b{0x123456789abcdef0ULL};
  SplitMix64 c{0x123456789abcdef1ULL};

  bool differs = false;
  for (int i = 0; i < 64; ++i) {
    const auto av = a.next_u64();
    const auto bv = b.next_u64();
    const auto cv = c.next_u64();
    assert(av == bv);
    differs |= av != cv;
  }
  assert(differs);
}

void test_uniform_bounded() {
  using namespace cugo::core;
  for (std::uint32_t bound = 2; bound <= 48; ++bound) {
    SplitMix64 rng{0x38f1c55f12340000ULL + bound};
    for (int i = 0; i < 4096; ++i) {
      assert(uniform_bounded(rng, bound) < bound);
    }
  }
}

void test_base_48_deal() {
  using namespace cugo::game;
  static_assert(kBaseStockCards == 20);
  static_assert(kInitialSampledCards == 28);

  const InitialDeal48 fixed = deal_base_48(0x123456789abcdef0ULL);
  assert(is_valid_initial_deal(fixed));
  assert(fixed.hand0 == 0x0000000520152490ULL);
  assert(fixed.hand1 == 0x000000909000d062ULL);
  assert(fixed.floor == 0x00000a080c600008ULL);
  assert(fixed.stock == 0x0000f562438a0b05ULL);
  assert(fixed.rng_state == 0x6045a6c286e2713cULL);

  for (std::uint64_t i = 0; i < 4096; ++i) {
    const auto deal = deal_base_48(cugo::core::derive_seed(0xc001d00d5eedULL, i));
    assert(is_valid_initial_deal(deal));
  }
}

}  // namespace

int main() {
  test_card_layout();
  test_pop_first_card();
  test_select_card_by_rank();
  test_rng_determinism();
  test_uniform_bounded();
  test_base_48_deal();
  std::cout << "cugo_core_test: PASS\n";
  return 0;
}
