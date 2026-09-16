#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"

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

}  // namespace

int main() {
  test_card_layout();
  test_pop_first_card();
  test_rng_determinism();
  std::cout << "cugo_core_test: PASS\n";
  return 0;
}
