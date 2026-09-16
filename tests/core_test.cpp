#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/deal.h"
#include "cugo/game/state.h"
#include "cugo/game/turn.h"

namespace {

bool same_turn_state(const cugo::game::TurnState48& a,
                     const cugo::game::TurnState48& b) {
  return a.hand0 == b.hand0 && a.hand1 == b.hand1 && a.floor == b.floor &&
         a.stock == b.stock && a.captured0 == b.captured0 &&
         a.captured1 == b.captured1 && a.rng_state == b.rng_state &&
         a.ppuk_months == b.ppuk_months &&
         a.ppuk_owner1_months == b.ppuk_owner1_months &&
         a.turn_index == b.turn_index && a.actor == b.actor &&
         a.phase == b.phase && a.pending_played == b.pending_played &&
         a.pending_drawn == b.pending_drawn;
}

cugo::game::TurnState48 make_resolve_fixture(
    cugo::core::CardMask floor,
    cugo::core::CardId played,
    cugo::core::CardId drawn,
    std::uint8_t actor = 0,
    std::uint16_t ppuk_months = 0,
    std::uint16_t ppuk_owner1_months = 0,
    bool final_stock_flip = false) {
  using namespace cugo::core;
  using namespace cugo::game;

  const CardMask pending = card_bit(played) | card_bit(drawn);
  assert((floor & pending) == 0);
  assert(played != drawn);
  const CardMask remaining = kFullDeckMask & ~(floor | pending);

  TurnState48 state{};
  state.floor = floor;
  if (final_stock_flip) {
    state.captured1 = remaining;
  } else {
    state.stock = remaining;
  }
  state.rng_state = 0x123456789abcdef0ULL;
  state.ppuk_months = ppuk_months;
  state.ppuk_owner1_months = ppuk_owner1_months;
  state.actor = actor;
  state.phase = TurnPhase::kResolve;
  state.pending_played = played;
  state.pending_drawn = drawn;
  assert(is_valid_turn_state(state));
  return state;
}

void test_card_layout() {
  using namespace cugo::core;
  static_assert(kCardCount == 48);
  static_assert(kFullDeckMask == 0x0000ffffffffffffULL);

  CardMask seen = 0;
  for (int month = 0; month < kMonthCount; ++month) {
    for (int slot = 0; slot < kCardsPerMonth; ++slot) {
      const CardId card = make_card(static_cast<std::uint8_t>(month),
                                    static_cast<std::uint8_t>(slot));
      assert(is_valid_card(card));
      assert(card_month(card) == month);
      assert(card_slot(card) == slot);
      seen |= card_bit(card);
    }
  }
  assert(seen == kFullDeckMask);
  assert(card_count(seen) == kCardCount);
}

void test_month_masks() {
  using namespace cugo::core;
  CardMask all = 0;
  for (std::uint8_t month = 0; month < kMonthCount; ++month) {
    const CardMask mask = month_mask(month);
    assert(card_count(mask) == kCardsPerMonth);
    assert((all & mask) == 0);
    all |= mask;
    for (std::uint8_t slot = 0; slot < kCardsPerMonth; ++slot) {
      const CardId card = make_card(month, slot);
      assert(matching_month_cards(kFullDeckMask, card) == mask);
    }
  }
  assert(all == kFullDeckMask);
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
  const CardMask mask =
      card_bit(0) | card_bit(7) | card_bit(31) | card_bit(32) | card_bit(47);
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
    const auto deal =
        deal_base_48(cugo::core::derive_seed(0xc001d00d5eedULL, i));
    assert(is_valid_initial_deal(deal));
  }
}

void test_resident_stock_draws() {
  using namespace cugo::core;
  using namespace cugo::game;

  constexpr std::uint64_t kMasterSeed = 0x73746174655f3031ULL;
  for (std::uint64_t game = 0; game < 4096; ++game) {
    const auto seed = derive_seed(kMasterSeed, game);
    const auto deal = deal_base_48(seed);
    auto state = make_resident_state(deal);
    const CardMask initial_stock = deal.stock;
    CardMask drawn = 0;

    for (int step = 0; step < kBaseStockCards; ++step) {
      const int before = card_count(state.stock);
      const CardId card = draw_stock_card(state);
      assert(card != kInvalidCard);
      assert((initial_stock & card_bit(card)) != 0);
      assert((drawn & card_bit(card)) == 0);
      drawn |= card_bit(card);
      assert(card_count(state.stock) == before - 1);
    }

    assert(state.stock == 0);
    assert(drawn == initial_stock);
    const std::uint64_t rng_before = state.rng_state;
    assert(draw_stock_card(state) == kInvalidCard);
    assert(state.rng_state == rng_before);
  }
}

void test_turn_phase_frame() {
  using namespace cugo::core;
  using namespace cugo::game;

  constexpr std::uint64_t kMasterSeed = 0x7475726e5f303031ULL;
  for (std::uint64_t game = 0; game < 4096; ++game) {
    const auto deal = deal_base_48(derive_seed(kMasterSeed, game));
    const std::uint8_t actor = static_cast<std::uint8_t>(game & 1u);
    auto state = make_turn_state(deal, actor);
    assert(is_valid_turn_state(state));
    assert(state.phase == TurnPhase::kPlay);
    assert(state.actor == actor);
    assert(state.captured0 == 0);
    assert(state.captured1 == 0);
    assert(state.ppuk_months == 0);
    assert(state.ppuk_owner1_months == 0);

    assert(begin_regular_play(state, kInvalidCard) == TurnStatus::kInvalidCard);
    assert(draw_for_turn(state) == TurnStatus::kWrongPhase);
    const CardMask other_hand = actor == 0 ? state.hand1 : state.hand0;
    assert(begin_regular_play(state, first_card(other_hand)) ==
           TurnStatus::kCardNotInHand);
    assert(is_valid_turn_state(state));

    const CardId played = first_card(active_hand(state));
    assert(played != kInvalidCard);
    assert(begin_regular_play(state, played) == TurnStatus::kOk);
    assert(state.phase == TurnPhase::kDraw);
    assert(is_valid_turn_state(state));

    assert(draw_for_turn(state) == TurnStatus::kOk);
    assert(state.phase == TurnPhase::kResolve);
    assert(is_valid_turn_state(state));
  }
}

void test_resolve_fixed_cases() {
  using namespace cugo::core;
  using namespace cugo::game;

  {
    auto state = make_resolve_fixture(card_bit(0) | card_bit(4), 1, 8);
    const auto result = resolve_turn(state);
    assert(result.status == ResolveStatus::kOk);
    assert(result.events == kResolveEventNone);
    assert(result.captured_cards == (card_bit(0) | card_bit(1)));
    assert(state.captured0 == result.captured_cards);
    assert(state.floor == (card_bit(4) | card_bit(8)));
    assert(state.phase == TurnPhase::kPlay && state.actor == 1 &&
           state.turn_index == 1);
    assert(is_valid_turn_state(state));
  }

  {
    auto state = make_resolve_fixture(card_bit(0) | card_bit(4), 1, 2);
    const auto result = resolve_turn(state);
    assert(result.status == ResolveStatus::kOk);
    assert((result.events & kResolveEventPpuk) != 0);
    assert(result.captured_cards == 0);
    assert(state.floor ==
           (card_bit(0) | card_bit(1) | card_bit(2) | card_bit(4)));
    assert((state.ppuk_months & 1u) != 0);
    assert((state.ppuk_owner1_months & 1u) == 0);
    assert(is_valid_turn_state(state));
  }

  {
    auto state = make_resolve_fixture(card_bit(4), 0, 1);
    const auto result = resolve_turn(state);
    assert(result.status == ResolveStatus::kOk);
    assert((result.events & kResolveEventJjok) != 0);
    assert(result.captured_cards == (card_bit(0) | card_bit(1)));
    assert(state.floor == card_bit(4));
    assert(is_valid_turn_state(state));
  }

  {
    auto state =
        make_resolve_fixture(card_bit(0) | card_bit(1) | card_bit(4), 2, 3);
    const auto result = resolve_turn(state);
    assert(result.status == ResolveStatus::kOk);
    assert((result.events & kResolveEventTtadak) != 0);
    assert(result.captured_cards == month_mask(0));
    assert(state.floor == card_bit(4));
    assert(is_valid_turn_state(state));
  }

  {
    auto state = make_resolve_fixture(card_bit(0) | card_bit(4), 1, 5);
    const auto result = resolve_turn(state);
    assert(result.status == ResolveStatus::kOk);
    assert((result.events & kResolveEventSweep) != 0);
    assert(result.captured_cards ==
           (card_bit(0) | card_bit(1) | card_bit(4) | card_bit(5)));
    assert(state.floor == 0);
    assert(is_valid_turn_state(state));
  }

  {
    const std::uint16_t month0 = 1u;
    auto state = make_resolve_fixture(
        card_bit(0) | card_bit(1) | card_bit(2) | card_bit(4),
        3,
        8,
        0,
        month0,
        month0);
    const auto result = resolve_turn(state);
    assert(result.status == ResolveStatus::kOk);
    assert(result.captured_opponent_ppuk == 1);
    assert(result.captured_own_ppuk == 0);
    assert((state.ppuk_months & month0) == 0);
    assert((state.ppuk_owner1_months & month0) == 0);
    assert((state.captured0 & month_mask(0)) == month_mask(0));
    assert(is_valid_turn_state(state));
  }
}

void test_resolve_choice_and_deferred_last_card() {
  using namespace cugo::core;
  using namespace cugo::game;

  auto state =
      make_resolve_fixture(card_bit(0) | card_bit(1) | card_bit(4), 2, 8);
  const auto before = state;
  auto result = resolve_turn(state);
  assert(result.status == ResolveStatus::kChoiceRequired);
  assert(same_turn_state(state, before));

  result = resolve_turn(state, ResolveChoices{4, kInvalidCard});
  assert(result.status == ResolveStatus::kInvalidChoice);
  assert(same_turn_state(state, before));

  result = resolve_turn(state, ResolveChoices{0, kInvalidCard});
  assert(result.status == ResolveStatus::kOk);
  assert(result.captured_cards == (card_bit(0) | card_bit(2)));
  assert(state.floor == (card_bit(1) | card_bit(4) | card_bit(8)));
  assert(is_valid_turn_state(state));

  auto drawn_choice =
      make_resolve_fixture(card_bit(0) | card_bit(4) | card_bit(5), 1, 6);
  const auto drawn_choice_before = drawn_choice;
  result = resolve_turn(drawn_choice);
  assert(result.status == ResolveStatus::kChoiceRequired);
  assert(result.captured_cards == 0);
  assert(result.events == kResolveEventNone);
  assert(same_turn_state(drawn_choice, drawn_choice_before));

  auto last_ppuk = make_resolve_fixture(card_bit(0), 1, 2, 0, 0, 0, true);
  const auto last_before = last_ppuk;
  result = resolve_turn(last_ppuk);
  assert(result.status == ResolveStatus::kUnsupportedLastCardSpecial);
  assert(same_turn_state(last_ppuk, last_before));

  auto last_jjok = make_resolve_fixture(card_bit(4), 0, 1, 0, 0, 0, true);
  const auto last_jjok_before = last_jjok;
  result = resolve_turn(last_jjok);
  assert(result.status == ResolveStatus::kUnsupportedLastCardSpecial);
  assert(same_turn_state(last_jjok, last_jjok_before));
}

void test_random_resolve_without_choices() {
  using namespace cugo::core;
  using namespace cugo::game;

  constexpr std::uint64_t kMasterSeed = 0x7265736f6c766531ULL;
  int resolved = 0;
  int choices = 0;
  for (std::uint64_t game = 0; game < 4096; ++game) {
    const auto deal = deal_base_48(derive_seed(kMasterSeed, game));
    auto state = make_turn_state(deal, static_cast<std::uint8_t>(game & 1u));
    const std::uint8_t original_actor = state.actor;
    const CardId played = first_card(active_hand(state));
    assert(begin_regular_play(state, played) == TurnStatus::kOk);
    assert(draw_for_turn(state) == TurnStatus::kOk);
    const auto before_resolve = state;

    const auto result = resolve_turn(state);
    if (result.status == ResolveStatus::kOk) {
      ++resolved;
      assert(state.phase == TurnPhase::kPlay);
      assert(state.actor == (original_actor ^ 1u));
      assert(state.turn_index == 1);
      assert(is_valid_turn_state(state));
    } else {
      assert(result.status == ResolveStatus::kChoiceRequired);
      ++choices;
      assert(same_turn_state(state, before_resolve));
      assert(is_valid_turn_state(state));
    }
  }
  assert(resolved > 0);
  assert(choices > 0);
}

}  // namespace

int main() {
  test_card_layout();
  test_month_masks();
  test_pop_first_card();
  test_select_card_by_rank();
  test_rng_determinism();
  test_uniform_bounded();
  test_base_48_deal();
  test_resident_stock_draws();
  test_turn_phase_frame();
  test_resolve_fixed_cases();
  test_resolve_choice_and_deferred_last_card();
  test_random_resolve_without_choices();
  std::cout << "cugo_core_test: PASS\n";
  return 0;
}
