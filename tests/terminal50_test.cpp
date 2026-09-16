#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "cugo/game/terminal50.h"

namespace {

using cugo::core::CardMask;
using namespace cugo::game;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

CardMask take_safe_cards(CardMask& remaining, CardMask current, int count) {
  CardMask added = 0;
  while (count-- > 0) {
    cugo::core::CardId selected = cugo::core::kInvalidCard;
    CardMask scan = remaining;
    while (scan != 0) {
      const cugo::core::CardId card = cugo::core::pop_first_card(scan);
      const CardMask bit = cugo::core::card_bit(card);
      const std::uint8_t month = cugo::core::card_month(card);
      if (cugo::core::card_count((current | added | bit) &
                                 cugo::core::month_mask(month)) < 4) {
        selected = card;
        break;
      }
    }
    if (selected == cugo::core::kInvalidCard) {
      selected = cugo::core::first_card(remaining);
    }
    const CardMask bit = cugo::core::card_bit(selected);
    remaining &= ~bit;
    added |= bit;
  }
  return added;
}

InitialDeal50 make_deal(CardMask hand0_fixed = 0,
                        CardMask hand1_fixed = 0,
                        CardMask floor_fixed = 0) {
  require((hand0_fixed & hand1_fixed) == 0 &&
              (hand0_fixed & floor_fixed) == 0 &&
              (hand1_fixed & floor_fixed) == 0,
          "fixed deal masks overlap");
  CardMask remaining = cugo::core::kShinMatgoDeckMask &
                       ~(hand0_fixed | hand1_fixed | floor_fixed);
  CardMask hand0 = hand0_fixed;
  CardMask hand1 = hand1_fixed;
  CardMask floor = floor_fixed;

  hand0 |= take_safe_cards(remaining, hand0,
                           kInitialHandCards - cugo::core::card_count(hand0));
  hand1 |= take_safe_cards(remaining, hand1,
                           kInitialHandCards - cugo::core::card_count(hand1));
  floor |= take_safe_cards(remaining, floor,
                           kInitialFloorCards - cugo::core::card_count(floor));
  InitialDeal50 deal{hand0, hand1, floor, remaining, 0, 0,
                     0x123456789abcdef0ull};
  require(is_valid_initial_deal(deal), "constructed deal must be valid");
  return deal;
}

void test_initial_chongtong() {
  {
    const InitialDeal50 deal = make_deal(cugo::core::month_mask(0));
    const TerminalGameState50 state = make_terminal_game_state50(deal, 0);
    require(terminal50_is_finished(state), "single hand chongtong must end game");
    require(state.terminal_reason == TerminalReason50::kInitialChongtong,
            "single hand chongtong reason");
    require(state.terminal_winner == 0 && state.terminal_points == 10,
            "single hand chongtong winner/points");
    require(is_valid_terminal_game_state50(state), "single hand terminal invariant");
  }
  {
    const InitialDeal50 deal = make_deal(cugo::core::month_mask(0),
                                         cugo::core::month_mask(1));
    const TerminalGameState50 state = make_terminal_game_state50(deal, 1);
    require(state.terminal_reason == TerminalReason50::kInitialChongtong,
            "double hand chongtong reason");
    require(state.terminal_winner == 1 && state.terminal_points == 10,
            "first player wins double hand chongtong");
    require(is_valid_terminal_game_state50(state), "double hand terminal invariant");
  }
  {
    const InitialDeal50 deal = make_deal(0, 0, cugo::core::month_mask(2));
    const TerminalGameState50 state = make_terminal_game_state50(deal, 0);
    require(state.terminal_reason == TerminalReason50::kFloorChongtong,
            "floor chongtong reason");
    require(state.terminal_winner == 0 && state.terminal_points == 10,
            "floor chongtong current convention");
    require(is_valid_terminal_game_state50(state), "floor terminal invariant");
  }
}

TerminalGameState50 make_minimal_bonus_chongtong_state() {
  TerminalGameState50 state{};
  state.special.game.turn.hand0 =
      cugo::core::card_bit(cugo::core::make_card(0, 0)) |
      cugo::core::card_bit(cugo::core::make_card(0, 1)) |
      cugo::core::card_bit(cugo::core::make_card(0, 2)) |
      cugo::core::card_bit(cugo::core::kBonusTwoPi);
  state.special.game.turn.stock =
      cugo::core::card_bit(cugo::core::make_card(0, 3));
  state.special.game.turn.actor = 0;
  state.special.game.turn.phase = Turn50Phase::kPlay;
  state.special.game.turn.pending_played = cugo::core::kInvalidCard;
  state.special.game.turn.pending_drawn = cugo::core::kInvalidCard;
  state.special.game.decision_actor = kNoGame50Player;
  state.special.game.winner = kNoGame50Player;
  state.pending_chongtong_actor = kNoTerminal50Player;
  state.terminal_winner = kNoTerminal50Player;
  state.terminal_reason = TerminalReason50::kNone;
  return state;
}

void test_bonus_replacement_chongtong_choice() {
  TerminalGameState50 state = make_minimal_bonus_chongtong_state();
  const TerminalBonus50Result result = play_bonus_for_terminal50_with_pi_transfer(
      state, cugo::core::kBonusTwoPi);
  require(result.bonus_result.play.status == Turn50Status::kOk,
          "bonus play should succeed");
  require(result.choice_opened == 1 && result.new_chongtong_months == 1u,
          "bonus replacement should open month-0 chongtong choice");
  require(terminal50_has_chongtong_choice(state), "choice must remain pending");

  TerminalGameState50 win_state = state;
  require(apply_chongtong_action50(win_state, ChongtongAction50::kWin) ==
              TerminalStatus50::kOk,
          "bonus chongtong win action");
  require(win_state.terminal_reason == TerminalReason50::kBonusChongtong &&
              win_state.terminal_winner == 0 &&
              win_state.terminal_points == kChongtongPoints50,
          "bonus chongtong win terminal");

  require(apply_chongtong_action50(state, ChongtongAction50::kContinue) ==
              TerminalStatus50::kOk,
          "bonus chongtong continue action");
  require(!terminal50_has_chongtong_choice(state) &&
              !terminal50_is_finished(state),
          "continue must resume game");
}

void test_three_ppuk() {
  {
    TerminalGameState50 state = make_terminal_game_state50(make_deal(), 0);
    require(!record_completed_turn_ppuk50(state, 0, true), "first ppuk");
    require(!record_completed_turn_ppuk50(state, 0, false), "streak reset 1");
    require(!record_completed_turn_ppuk50(state, 0, true), "second ppuk");
    require(!record_completed_turn_ppuk50(state, 0, false), "streak reset 2");
    require(record_completed_turn_ppuk50(state, 0, true), "third ppuk terminal");
    require(state.terminal_reason == TerminalReason50::kThreePpuk &&
                state.terminal_winner == 0 &&
                state.terminal_points == kThreePpukPoints50,
            "nonconsecutive three-ppuk score");
    require(state.ppuk_count0 == 3 && state.ppuk_streak0 == 1,
            "nonconsecutive ppuk counters");
    require(is_valid_terminal_game_state50(state), "three-ppuk invariant");
  }
  {
    TerminalGameState50 state = make_terminal_game_state50(make_deal(), 1);
    require(!record_completed_turn_ppuk50(state, 1, true), "consecutive ppuk 1");
    require(!record_completed_turn_ppuk50(state, 1, true), "consecutive ppuk 2");
    require(record_completed_turn_ppuk50(state, 1, true), "consecutive ppuk 3");
    require(state.terminal_reason == TerminalReason50::kThreePpuk &&
                state.terminal_winner == 1 &&
                state.terminal_points == kThreeConsecutivePpukPoints50,
            "consecutive three-ppuk score");
    require(state.ppuk_count1 == 3 && state.ppuk_streak1 == 3,
            "consecutive ppuk counters");
    require(is_valid_terminal_game_state50(state), "49-point ppuk invariant");
  }
}

}  // namespace

int main() {
  try {
    test_initial_chongtong();
    test_bonus_replacement_chongtong_choice();
    test_three_ppuk();
    std::cout << "cugo_terminal50_test: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cugo_terminal50_test: FAIL: " << e.what() << '\n';
    return 1;
  }
}
