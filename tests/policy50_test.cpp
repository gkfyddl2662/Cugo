#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/game/policy50.h"

namespace {

using namespace cugo::game;

bool same_action(const Action50& a, const Action50& b) {
  return a.kind == b.kind && a.card == b.card && a.month == b.month &&
         a.resolve_choices.played_match == b.resolve_choices.played_match &&
         a.resolve_choices.drawn_match == b.resolve_choices.drawn_match &&
         a.pi_selection.cards == b.pi_selection.cards;
}

TerminalGameState50 make_nonterminal(std::uint64_t seed) {
  for (std::uint64_t i = 0; i < 10000; ++i) {
    TerminalGameState50 state =
        make_terminal_game_state50(deal_shin_matgo_50(seed + i), 0);
    if (!terminal50_is_finished(state)) return state;
  }
  assert(false);
  return TerminalGameState50{};
}

void test_hidden_information_invariance() {
  TerminalGameState50 a = make_nonterminal(0x504f4c4943590000ULL);
  TerminalGameState50 b = a;

  auto& t = b.special.game.turn;
  const auto opponent_card = cugo::core::first_card(t.hand1);
  const auto stock_card = cugo::core::first_card(t.stock);
  assert(cugo::core::is_physical_card(opponent_card));
  assert(cugo::core::is_physical_card(stock_card));

  t.hand1 &= ~cugo::core::card_bit(opponent_card);
  t.stock &= ~cugo::core::card_bit(stock_card);
  t.hand1 |= cugo::core::card_bit(stock_card);
  t.stock |= cugo::core::card_bit(opponent_card);

  const PolicyPacket50 pa = make_policy_packet50(a);
  const PolicyPacket50 pb = make_policy_packet50(b);
  assert(equal_policy_packet50(pa, pb));
}

void test_legal_actions_apply() {
  std::uint64_t checked = 0;
  for (std::uint64_t seed = 1; seed <= 512; ++seed) {
    TerminalGameState50 state = make_terminal_game_state50(
        deal_shin_matgo_50(0x4c4547414c000000ULL ^ seed),
        static_cast<std::uint8_t>(seed & 1u));
    if (terminal50_is_finished(state)) continue;

    const PolicyActionMask50 legal = legal_primary_actions50(state);
    for (std::uint16_t index = 0; index < kPolicyActionCount50; ++index) {
      if (!has_policy_action50(legal, index)) continue;
      Action50 action = kInvalidAction50;
      assert(decode_primary_action50(state, index, action) == ActionStatus50::kOk);
      assert(primary_action_index50(action) == index);
      TerminalGameState50 copy = state;
      assert(apply_action50(copy, action).status == ActionStatus50::kOk);
      ++checked;
    }
  }
  assert(checked != 0);
}

void test_rollout_policy_roundtrip() {
  std::uint64_t decisions = 0;
  std::uint64_t legal_total = 0;
  std::uint64_t terminal = 0;
  std::uint64_t nagari = 0;

  for (std::uint32_t i = 0; i < 16384; ++i) {
    TerminalGameState50 state = make_terminal_game_state50(
        deal_shin_matgo_50(0x504f4c4943595000ULL + i),
        static_cast<std::uint8_t>(i & 1u));

    for (std::uint16_t step = 0; step < 128; ++step) {
      if (terminal50_is_finished(state)) {
        ++terminal;
        break;
      }
      if (!terminal50_has_chongtong_choice(state) &&
          !game50_has_pending_decision(state.special.game) &&
          rollout_hands_exhausted50(state)) {
        ++nagari;
        break;
      }

      const PolicyPacket50 packet = make_policy_packet50(state);
      const CanonicalAction50 canonical = canonical_action50(state);
      assert(canonical.status == ActionStatus50::kOk);
      const std::uint16_t index = primary_action_index50(canonical.action);
      assert(index < kPolicyActionCount50);
      assert(has_policy_action50(packet.legal_actions, index));
      legal_total += policy_action_count50(packet.legal_actions);

      Action50 decoded = kInvalidAction50;
      assert(decode_primary_action50(state, index, decoded) == ActionStatus50::kOk);
      assert(same_action(decoded, canonical.action));
      assert(apply_action50(state, decoded).status == ActionStatus50::kOk);
      ++decisions;
    }
  }

  assert(decisions != 0);
  assert(terminal + nagari == 16384);
  std::cout << "cugo_policy50_test: PASS (decisions=" << decisions
            << " terminal=" << terminal
            << " nagari=" << nagari
            << " avg_legal="
            << static_cast<double>(legal_total) / static_cast<double>(decisions)
            << " packet_bytes=" << sizeof(PolicyPacket50) << ")\n";
}

}  // namespace

int main() {
  static_assert(kPolicyActionCount50 == 127);
  static_assert(sizeof(PolicyActionMask50) == 16);
  test_hidden_information_invariance();
  test_legal_actions_apply();
  test_rollout_policy_roundtrip();
  return 0;
}
