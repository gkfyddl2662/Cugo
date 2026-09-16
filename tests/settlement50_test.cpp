#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "cugo/game/settlement50.h"

namespace {

using cugo::core::CardMask;
using namespace cugo::game;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

CardMask first_n(CardMask cards, int count) {
  CardMask out = 0;
  while (count-- > 0 && cards != 0) {
    const auto card = cugo::core::pop_first_card(cards);
    out |= cugo::core::card_bit(card);
  }
  return out;
}

TerminalGameState50 blank_stop_state(std::uint8_t winner = 0) {
  TerminalGameState50 state{};
  state.terminal_winner = static_cast<std::uint8_t>(winner & 1u);
  state.terminal_reason = TerminalReason50::kStop;
  return state;
}

void set_captured(TerminalGameState50& state,
                  std::uint8_t player,
                  CardMask cards) {
  if ((player & 1u) == 0) state.special.game.turn.captured0 = cards;
  else state.special.game.turn.captured1 = cards;
}

void test_unfinished() {
  TerminalGameState50 state{};
  state.terminal_winner = kNoTerminal50Player;
  state.terminal_reason = TerminalReason50::kNone;
  const Settlement50 result = settle_terminal50(state);
  require(result.status == SettlementStatus50::kGameNotFinished,
          "unfinished game must not settle");
  require(settlement_reward50(result, 0) == 0,
          "unfinished reward must be zero");
}

void test_plain_stop() {
  TerminalGameState50 state = blank_stop_state(0);
  set_captured(state, 0, kGodoriMask | kHongdanMask);
  set_captured(state, 1, first_n(kPlainPiMask, 8));

  const ScoreBreakdown score = persistent_score_player50(state.special.game.turn, 0);
  require(score.total_points >= kShinMatgoDecisionBasePoints,
          "plain stop fixture must be a legal winning score");

  const Settlement50 result = settle_terminal50(state);
  require(result.status == SettlementStatus50::kOk, "plain stop settles");
  require(result.flags == kSettlementNone, "plain stop has no doubles");
  require(result.go_multiplier == 1 && result.bomb_shake_multiplier == 1 &&
              result.bak_multiplier == 1,
          "plain stop multipliers");
  require(result.final_points == score.total_points,
          "plain stop final points");
}

void test_all_stop_multipliers() {
  TerminalGameState50 state = blank_stop_state(0);
  const CardMask winner_cards = first_n(kAnimalMask, 7) |
                                first_n(kBrightMask, 3) |
                                first_n(kPlainPiMask, 10);
  const CardMask loser_cards = first_n(kPlainPiMask & ~winner_cards, 7);
  set_captured(state, 0, winner_cards);
  set_captured(state, 1, loser_cards);

  state.special.game.go_count0 = 3;
  state.special.game.go_count1 = 1;
  state.special.bombs0 = 1;
  state.special.shaken0 = 1u;

  const ScoreBreakdown score = persistent_score_player50(state.special.game.turn, 0);
  const GoAdjustedScore50 go = go_adjusted_score50(score.total_points, 3);
  const Settlement50 result = settle_terminal50(state);
  const std::uint16_t expected_flags =
      kSettlementMeongtta | kSettlementPiBak |
      kSettlementGwangBak | kSettlementDokBak;

  require(result.status == SettlementStatus50::kOk, "combined stop settles");
  require(result.flags == expected_flags, "all four bak flags must apply");
  require(result.go_multiplier == 2, "3-go multiplier");
  require(result.bomb_shake_multiplier == 4,
          "one bomb plus one shake is x4");
  require(result.bak_multiplier == 16, "four independent doubles are x16");
  require(result.points_after_go == go.points_after_go,
          "go points must match game layer");
  require(result.final_points == go.points_after_go * 4u * 16u,
          "combined final points");
  require(settlement_reward50(result, 0) ==
              static_cast<std::int64_t>(result.final_points) &&
              settlement_reward50(result, 1) ==
              -static_cast<std::int64_t>(result.final_points),
          "zero-sum terminal rewards");
}

void test_fixed_special_terminal() {
  TerminalGameState50 state = blank_stop_state(1);
  state.terminal_reason = TerminalReason50::kThreePpuk;
  state.terminal_points = kThreeConsecutivePpukPoints50;
  state.special.game.go_count1 = 5;
  state.special.game.go_count0 = 1;
  state.special.bombs1 = 2;
  state.special.shaken1 = 3u;
  set_captured(state, 1, first_n(kAnimalMask, 7) |
                          first_n(kBrightMask, 3) |
                          first_n(kPlainPiMask, 10));

  const Settlement50 result = settle_terminal50(state);
  require(result.status == SettlementStatus50::kOk, "fixed terminal settles");
  require(result.flags == kSettlementFixedSpecial,
          "fixed terminal must stay isolated from regular doubles");
  require(result.final_points == kThreeConsecutivePpukPoints50,
          "49-point three-ppuk remains fixed");
  require(result.go_multiplier == 1 && result.bomb_shake_multiplier == 1 &&
              result.bak_multiplier == 1,
          "fixed terminal does not stack undocumented multipliers");
}

}  // namespace

int main() {
  try {
    test_unfinished();
    test_plain_stop();
    test_all_stop_multipliers();
    test_fixed_special_terminal();
    std::cout << "cugo_settlement50_test: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cugo_settlement50_test: FAIL: " << e.what() << '\n';
    return 1;
  }
}
