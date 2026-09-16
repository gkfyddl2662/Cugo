#include <cassert>
#include <cstdint>
#include <iostream>

#include "cugo/core/card.h"
#include "cugo/game/pi_transfer.h"

namespace {

void test_persistent_role_and_scoring() {
  using namespace cugo::core;
  using namespace cugo::game;

  TurnState50 state{};
  state.captured0 = card_bit(kGukjin);
  state.stock = kShinMatgoDeckMask & ~card_bit(kGukjin);
  state.phase = Turn50Phase::kPlay;
  state.pending_played = kInvalidCard;
  state.pending_drawn = kInvalidCard;

  assert(is_valid_persistent_turn_state50(state));
  assert(persistent_gukjin_role50(state, 0) == PersistentGukjinRole50::kAnimal);
  auto score = persistent_score_player50(state, 0);
  assert(score.animal_count == 1);
  assert(score.pi_units == 0);

  assert(set_persistent_gukjin_role50(state, 1, PersistentGukjinRole50::kDoublePi) ==
         PersistentGukjinRole50Status::kGukjinNotCaptured);
  assert(set_persistent_gukjin_role50(state, 2, PersistentGukjinRole50::kDoublePi) ==
         PersistentGukjinRole50Status::kInvalidPlayer);
  assert(set_persistent_gukjin_role50(
             state, 0, static_cast<PersistentGukjinRole50>(2)) ==
         PersistentGukjinRole50Status::kInvalidRole);

  assert(set_persistent_gukjin_role50(state, 0, PersistentGukjinRole50::kDoublePi) ==
         PersistentGukjinRole50Status::kOk);
  assert((state.ppuk_owner1_months & kPersistentGukjinPlayer0Flag50) != 0);
  assert(is_valid_persistent_turn_state50(state));
  score = persistent_score_player50(state, 0);
  assert(score.animal_count == 0);
  assert(score.pi_units == 2);

  assert(set_persistent_gukjin_role50(state, 0, PersistentGukjinRole50::kAnimal) ==
         PersistentGukjinRole50Status::kOk);
  assert((state.ppuk_owner1_months & kPersistentGukjinFlags50) == 0);
  assert(is_valid_persistent_turn_state50(state));
}

void test_role_moves_with_pi_transfer() {
  using namespace cugo::core;
  using namespace cugo::game;

  TurnState50 state{};
  state.captured1 = card_bit(kGukjin);
  assert(set_persistent_gukjin_role50(state, 1, PersistentGukjinRole50::kDoublePi) ==
         PersistentGukjinRole50Status::kOk);

  const auto transfer = apply_pi_steal50(state, 0, 1);
  assert(transfer.status == PiTransferStatus::kOk);
  assert(transfer.transferred_mask == card_bit(kGukjin));
  assert(transfer.transferred_pi_units == 2);
  assert(state.captured0 == card_bit(kGukjin));
  assert(state.captured1 == 0);
  assert(persistent_gukjin_role50(state, 0) == PersistentGukjinRole50::kDoublePi);
  assert(persistent_gukjin_role50(state, 1) == PersistentGukjinRole50::kAnimal);
  assert(persistent_score_player50(state, 0).pi_units == 2);

  assert(set_persistent_gukjin_role50(state, 0, PersistentGukjinRole50::kAnimal) ==
         PersistentGukjinRole50Status::kOk);
  assert(persistent_score_player50(state, 0).animal_count == 1);
  assert(persistent_score_player50(state, 0).pi_units == 0);
}

void test_packed_flag_survives_ppuk_metadata_updates() {
  using namespace cugo::core;
  using namespace cugo::game;

  TurnState50 state{};
  state.captured0 = card_bit(kGukjin);
  state.stock = kShinMatgoDeckMask & ~card_bit(kGukjin);
  state.phase = Turn50Phase::kPlay;
  state.pending_played = kInvalidCard;
  state.pending_drawn = kInvalidCard;
  assert(set_persistent_gukjin_role50(state, 0, PersistentGukjinRole50::kDoublePi) ==
         PersistentGukjinRole50Status::kOk);

  state.ppuk_months = 1u << 5;
  state.ppuk_owner1_months |= 1u << 5;
  state.floor = month_mask(5) & ~card_bit(make_card(5, 3));
  state.stock &= ~state.floor;
  assert(is_valid_persistent_turn_state50(state));

  clear_ppuk_month50(state, 1u << 5);
  assert((state.ppuk_owner1_months & kPersistentGukjinPlayer0Flag50) != 0);
  assert(persistent_gukjin_role50(state, 0) == PersistentGukjinRole50::kDoublePi);
}

}  // namespace

int main() {
  test_persistent_role_and_scoring();
  test_role_moves_with_pi_transfer();
  test_packed_flag_survives_ppuk_metadata_updates();
  std::cout << "cugo_gukjin_state_test: PASS\n";
  return 0;
}
