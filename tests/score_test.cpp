#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/hwatu.h"

namespace {

void test_primary_card_classes() {
  using namespace cugo::core;
  using namespace cugo::game;

  assert(card_count(kBrightMask) == 5);
  assert(card_count(kAnimalMask) == 9);
  assert(card_count(kRibbonMask) == 10);
  assert(card_count(kPlainPiMask) == 22);
  assert(card_count(kFixedDoublePiMask) == 2);
  assert(kPrimaryClassMask == kFullDeckMask);

  for (std::uint8_t month = 0; month < kMonthCount; ++month) {
    const CardMask primary = month_mask(month) & kPrimaryClassMask;
    assert(card_count(primary) == 4);
  }

  assert(kRainBright == make_card(11, 0));
  assert(kGukjin == make_card(8, 0));
  assert((kGodoriMask & kAnimalMask) == kGodoriMask);
  assert((kHongdanMask & kRibbonMask) == kHongdanMask);
  assert((kChodanMask & kRibbonMask) == kChodanMask);
  assert((kCheongdanMask & kRibbonMask) == kCheongdanMask);
}

void test_bright_scoring() {
  using namespace cugo::core;
  using namespace cugo::game;

  const CardMask three_without_rain =
      card_bit(make_card(0, 0)) | card_bit(make_card(2, 0)) |
      card_bit(make_card(7, 0));
  assert(score_captured(three_without_rain).bright_points == 3);

  const CardMask three_with_rain =
      card_bit(make_card(0, 0)) | card_bit(make_card(2, 0)) |
      card_bit(kRainBright);
  assert(score_captured(three_with_rain).bright_points == 2);

  CardMask four = kBrightMask & ~card_bit(kRainBright);
  assert(score_captured(four).bright_points == 4);
  assert(score_captured(kBrightMask).bright_points == 15);
}

void test_animal_ribbon_and_pi_scoring() {
  using namespace cugo::core;
  using namespace cugo::game;

  const auto godori = score_captured(kGodoriMask);
  assert(godori.animal_count == 3);
  assert(godori.animal_points == 5);
  assert((godori.flags & kScoreFlagGodori) != 0);

  const auto all_animals = score_captured(kAnimalMask);
  assert(all_animals.animal_count == 9);
  assert(all_animals.animal_points == 10);
  assert((all_animals.flags & kScoreFlagMeongtta) != 0);

  const auto all_ribbons = score_captured(kRibbonMask);
  assert(all_ribbons.ribbon_count == 10);
  assert(all_ribbons.ribbon_points == 15);
  assert((all_ribbons.flags & kScoreFlagHongdan) != 0);
  assert((all_ribbons.flags & kScoreFlagChodan) != 0);
  assert((all_ribbons.flags & kScoreFlagCheongdan) != 0);

  CardMask ten_plain_pi = 0;
  for (std::uint8_t month = 0; month < 5; ++month) {
    ten_plain_pi |= card_bit(make_card(month, 2));
    ten_plain_pi |= card_bit(make_card(month, 3));
  }
  assert(pi_units(ten_plain_pi) == 10);
  assert(score_captured(ten_plain_pi).pi_points == 1);

  const CardMask fixed_double = kFixedDoublePiMask;
  assert(pi_units(fixed_double) == 4);
  assert(pi_card_mask(fixed_double) == fixed_double);
}

void test_gukjin_dual_use() {
  using namespace cugo::core;
  using namespace cugo::game;

  const CardMask gukjin = card_bit(kGukjin);
  const auto as_animal = score_captured(gukjin);
  assert(as_animal.animal_count == 1);
  assert(as_animal.pi_units == 0);
  assert(pi_card_mask(gukjin) == 0);

  const ScoreOptions as_pi{true};
  const auto converted = score_captured(gukjin, as_pi);
  assert(converted.animal_count == 0);
  assert(converted.pi_units == 2);
  assert(pi_card_mask(gukjin, as_pi) == gukjin);

  const auto full_default = score_captured(kFullDeckMask);
  const auto full_converted = score_captured(kFullDeckMask, as_pi);
  assert(full_default.pi_units == 26);
  assert(full_default.total_points == 57);
  assert(full_converted.pi_units == 28);
  assert(full_converted.total_points == 58);
}

void test_random_score_invariants() {
  using namespace cugo::core;
  using namespace cugo::game;

  SplitMix64 rng{0x73636f72655f3031ULL};
  for (int i = 0; i < 4096; ++i) {
    const CardMask cards = rng.next_u64() & kFullDeckMask;
    const ScoreOptions options{(i & 1) != 0};
    const auto score = score_captured(cards, options);
    assert(score.bright_count <= 5);
    assert(score.animal_count <= 9);
    assert(score.ribbon_count <= 10);
    assert(score.pi_units <= 28);
    assert(score.total_points ==
           static_cast<unsigned>(score.bright_points) + score.animal_points +
               score.ribbon_points + score.pi_points);
    assert((pi_card_mask(cards, options) & ~cards) == 0);
  }
}

}  // namespace

int main() {
  test_primary_card_classes();
  test_bright_scoring();
  test_animal_ribbon_and_pi_scoring();
  test_gukjin_dual_use();
  test_random_score_invariants();
  std::cout << "cugo_score_test: PASS\n";
  return 0;
}
