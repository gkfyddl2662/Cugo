#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/game/game50.h"
#include "cugo/game/hwatu.h"

namespace {

bool same_turn(const cugo::game::TurnState50& a,
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

bool same_game(const cugo::game::GameState50& a,
               const cugo::game::GameState50& b) {
  return same_turn(a.turn, b.turn) &&
         a.go_count0 == b.go_count0 && a.go_count1 == b.go_count1 &&
         a.last_go_base_score0 == b.last_go_base_score0 &&
         a.last_go_base_score1 == b.last_go_base_score1 &&
         a.decision_actor == b.decision_actor && a.winner == b.winner;
}

cugo::core::CardMask score8_cards() {
  return cugo::game::kGodoriMask | cugo::game::kHongdanMask;
}

cugo::core::CardMask score9_cards() {
  return score8_cards() |
         cugo::core::card_bit(cugo::core::make_card(3, 1)) |
         cugo::core::card_bit(cugo::core::make_card(4, 1));
}

cugo::core::CardMask score10_cards() {
  return score9_cards() |
         cugo::core::card_bit(cugo::core::make_card(5, 1));
}

cugo::game::GameState50 make_scored_game(std::uint8_t player,
                                         cugo::core::CardMask captured) {
  using namespace cugo::core;
  using namespace cugo::game;
  GameState50 state{};
  if (player == 0) state.turn.captured0 = captured;
  else state.turn.captured1 = captured;
  state.turn.stock = kShinMatgoDeckMask & ~captured;
  state.turn.actor = static_cast<std::uint8_t>(player ^ 1u);
  state.turn.phase = Turn50Phase::kPlay;
  state.turn.pending_played = kInvalidCard;
  state.turn.pending_drawn = kInvalidCard;
  state.decision_actor = kNoGame50Player;
  state.winner = kNoGame50Player;
  assert(is_valid_game_state50(state));
  return state;
}

void replace_captured(cugo::game::GameState50& state,
                      std::uint8_t player,
                      cugo::core::CardMask captured) {
  using namespace cugo::core;
  state.turn.captured0 = player == 0 ? captured : 0;
  state.turn.captured1 = player == 1 ? captured : 0;
  state.turn.stock = kShinMatgoDeckMask & ~captured;
}

void test_go_score_formula() {
  using cugo::game::go_adjusted_score50;
  assert(go_adjusted_score50(7, 0).points_after_go == 7);
  assert(go_adjusted_score50(7, 1).points_after_go == 8);
  assert(go_adjusted_score50(7, 2).points_after_go == 9);
  assert(go_adjusted_score50(7, 3).points_after_go == 20);
  assert(go_adjusted_score50(7, 4).points_after_go == 44);
  assert(go_adjusted_score50(7, 5).points_after_go == 96);
  assert(go_adjusted_score50(11, 3).go_multiplier == 2);
  assert(go_adjusted_score50(11, 4).go_multiplier == 4);
}

void test_first_decision_go_and_additional_score_gate() {
  using namespace cugo::game;
  auto state = make_scored_game(0, score8_cards());
  assert(game50_base_score(state, 0) == 8);
  assert(next_go_stop_threshold50(state, 0) == 7);
  assert(maybe_open_go_stop_decision50(state, 0));
  assert(state.decision_actor == 0);
  assert(is_valid_game_state50(state));

  assert(apply_go_stop_decision50(state, GoStopAction50::kGo) ==
         GoStopStatus50::kOk);
  assert(state.go_count0 == 1);
  assert(state.last_go_base_score0 == 8);
  assert(next_go_stop_threshold50(state, 0) == 9);
  assert(!game50_has_pending_decision(state));
  assert(is_valid_game_state50(state));

  assert(!maybe_open_go_stop_decision50(state, 0));
  replace_captured(state, 0, score9_cards());
  assert(game50_base_score(state, 0) == 9);
  assert(maybe_open_go_stop_decision50(state, 0));
  assert(apply_go_stop_decision50(state, GoStopAction50::kGo) ==
         GoStopStatus50::kOk);
  assert(state.go_count0 == 2);
  assert(state.last_go_base_score0 == 9);

  replace_captured(state, 0, score10_cards());
  assert(maybe_open_go_stop_decision50(state, 0));
  assert(apply_go_stop_decision50(state, GoStopAction50::kGo) ==
         GoStopStatus50::kOk);
  assert(state.go_count0 == 3);
  const auto adjusted = go_adjusted_score_player50(state, 0);
  assert(adjusted.base_points == 10);
  assert(adjusted.additive_go_points == 3);
  assert(adjusted.go_multiplier == 2);
  assert(adjusted.points_after_go == 26);
  assert(is_valid_game_state50(state));
}

void test_stop_finishes_and_blocks_turn_actions() {
  using namespace cugo::core;
  using namespace cugo::game;
  auto state = make_scored_game(1, score8_cards());
  assert(maybe_open_go_stop_decision50(state, 1));
  assert(apply_go_stop_decision50(state, GoStopAction50::kStop) ==
         GoStopStatus50::kOk);
  assert(game50_is_finished(state));
  assert(state.winner == 1);
  assert(!game50_has_pending_decision(state));
  assert(begin_regular_play_game50(state, kInvalidCard) ==
         Turn50Status::kWrongPhase);
  assert(apply_go_stop_decision50(state, GoStopAction50::kGo) ==
         GoStopStatus50::kGameFinished);
  assert(is_valid_game_state50(state));
}

void test_resolve_opens_decision_after_committed_turn() {
  using namespace cugo::core;
  using namespace cugo::game;
  const CardMask captured = score8_cards();
  const CardId floor_card = make_card(6, 2);
  const CardId played = make_card(10, 0);
  const CardId drawn = make_card(11, 1);

  GameState50 state{};
  state.turn.captured0 = captured;
  state.turn.floor = card_bit(floor_card);
  state.turn.pending_played = played;
  state.turn.pending_drawn = drawn;
  state.turn.stock = kShinMatgoDeckMask &
                     ~(captured | state.turn.floor | card_bit(played) |
                       card_bit(drawn));
  state.turn.actor = 0;
  state.turn.phase = Turn50Phase::kResolve;
  state.decision_actor = kNoGame50Player;
  state.winner = kNoGame50Player;
  assert(is_valid_game_state50(state));

  const auto result = resolve_game_turn50_with_pi_transfer(state);
  assert(result.turn_result.resolve.status == Resolve50Status::kOk);
  assert(result.turn_result.pi_transfer.status == PiTransferStatus::kOk);
  assert(result.completed_actor == 0);
  assert(result.base_score == 8);
  assert(result.decision_opened == 1);
  assert(state.decision_actor == 0);
  assert(state.turn.actor == 1);
  assert(is_valid_game_state50(state));
}

void test_pi_selection_keeps_whole_game_transactional() {
  using namespace cugo::core;
  using namespace cugo::game;
  CardMask pi = kPlainPiMask;
  const CardId first = pop_first_card(pi);
  const CardId second = pop_first_card(pi);
  const CardId played = make_card(0, 0);
  const CardId drawn = make_card(0, 1);
  const CardId floor_card = make_card(1, 0);

  GameState50 state{};
  state.turn.actor = 0;
  state.turn.phase = Turn50Phase::kResolve;
  state.turn.floor = card_bit(floor_card);
  state.turn.captured1 = card_bit(first) | card_bit(second);
  state.turn.pending_played = played;
  state.turn.pending_drawn = drawn;
  state.turn.stock = kShinMatgoDeckMask &
                     ~(state.turn.floor | state.turn.captured1 |
                       card_bit(played) | card_bit(drawn));
  state.decision_actor = kNoGame50Player;
  state.winner = kNoGame50Player;
  assert(is_valid_game_state50(state));
  const auto before = state;

  const auto result = resolve_game_turn50_with_pi_transfer(state);
  assert(result.turn_result.resolve.status == Resolve50Status::kOk);
  assert(result.turn_result.pi_transfer.status ==
         PiTransferStatus::kSelectionRequired);
  assert(result.decision_opened == 0);
  assert(same_game(state, before));
}

}  // namespace

int main() {
  test_go_score_formula();
  test_first_decision_go_and_additional_score_gate();
  test_stop_finishes_and_blocks_turn_actions();
  test_resolve_opens_decision_after_committed_turn();
  test_pi_selection_keeps_whole_game_transactional();
  std::cout << "cugo_game50_test: PASS\n";
  return 0;
}
