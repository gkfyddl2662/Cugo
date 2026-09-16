#include <cstdint>
#include <iostream>
#include "cugo/core/card.h"
#include "cugo/game/special50.h"

#define CHECK(x) do { if (!(x)) { std::cerr << "special50 check failed line " << __LINE__ << '\n'; return false; } } while (false)

namespace {
using namespace cugo::core;
using namespace cugo::game;

SpecialGameState50 base() {
  SpecialGameState50 s{};
  s.game.turn.rng_state = 0x123456789abcdef0ULL;
  s.game.turn.phase = Turn50Phase::kPlay;
  s.game.turn.pending_played = kInvalidCard;
  s.game.turn.pending_drawn = kInvalidCard;
  s.game.decision_actor = kNoGame50Player;
  s.game.winner = kNoGame50Player;
  return s;
}

bool shake_case() {
  auto s = base();
  s.game.turn.hand0 = card_bit(0) | card_bit(1) | card_bit(2) | card_bit(4) | card_bit(5);
  s.game.turn.floor = card_bit(6) | card_bit(7);
  s.game.turn.stock = kShinMatgoDeckMask & ~(s.game.turn.hand0 | s.game.turn.floor);
  CHECK((legal_shake_months50(s) & 1u) != 0);
  CHECK((legal_grenade_months50(s) & (1u << 1)) != 0);
  const SpecialAction50 a{SpecialKind50::kShake, make_card(0, 0), 0, kInvalidCard,
                          kNoPiTransferSelection};
  const auto r = apply_special_action50(s, a);
  CHECK(r.status == SpecialStatus50::kOk);
  CHECK(s.game.turn.phase == Turn50Phase::kDraw);
  CHECK((s.shaken0 & 1u) != 0);
  CHECK(bomb_shake_multiplier50(s, 0) == 2);
  return true;
}

bool bomb_case() {
  auto s = base();
  s.game.turn.hand0 = card_bit(0) | card_bit(1) | card_bit(2);
  s.game.turn.floor = card_bit(3) | card_bit(8);
  s.game.turn.captured1 = card_bit(10);
  s.game.turn.stock = card_bit(12);
  const auto r = play_bomb50(s, 0);
  const CardMask four = card_bit(0) | card_bit(1) | card_bit(2) | card_bit(3);
  CHECK(r.status == SpecialStatus50::kOk && (r.events & kSpecialBomb) != 0);
  CHECK(s.bombs0 == 1 && s.credits0 == 2 && bomb_shake_multiplier50(s, 0) == 2);
  CHECK(s.game.turn.actor == 1 && s.game.turn.turn_index == 1);
  CHECK((s.game.turn.captured0 & four) == four);
  CHECK((s.game.turn.captured0 & card_bit(10)) != 0 && (s.game.turn.captured1 & card_bit(10)) == 0);
  return true;
}

bool grenade_case() {
  auto s = base();
  s.game.turn.hand0 = card_bit(4) | card_bit(5);
  s.game.turn.floor = card_bit(6) | card_bit(7) | card_bit(8);
  s.game.turn.captured1 = card_bit(10);
  s.game.turn.stock = card_bit(12);
  const auto r = play_grenade50(s, 1);
  CHECK(r.status == SpecialStatus50::kOk && (r.events & kSpecialGrenade) != 0);
  CHECK(s.bombs0 == 0 && s.credits0 == 1 && bomb_shake_multiplier50(s, 0) == 1);
  CHECK((s.game.turn.captured0 & card_bit(10)) != 0);
  return true;
}

bool credit_case() {
  auto s = base();
  s.credits0 = 1;
  s.game.turn.hand0 = card_bit(0);
  s.game.turn.floor = card_bit(4);
  s.game.turn.stock = card_bit(8);
  const CardMask hand = s.game.turn.hand0;
  const auto r = play_bomb_credit50(s);
  CHECK(r.status == SpecialStatus50::kOk && (r.events & kSpecialBombCredit) != 0);
  CHECK(s.credits0 == 0 && s.game.turn.hand0 == hand && s.game.turn.actor == 1);
  return true;
}

bool rollback_case() {
  auto s = base();
  s.game.turn.hand0 = card_bit(0) | card_bit(1) | card_bit(2);
  s.game.turn.floor = card_bit(3) | card_bit(8);
  s.game.turn.captured1 = card_bit(10) | card_bit(14);
  s.game.turn.stock = card_bit(12);
  const auto before = s;
  const auto r = play_bomb50(s, 0);
  CHECK(r.status == SpecialStatus50::kPiSelectionRequired);
  CHECK(s.game.turn.hand0 == before.game.turn.hand0 && s.game.turn.floor == before.game.turn.floor);
  CHECK(s.game.turn.stock == before.game.turn.stock && s.game.turn.captured0 == before.game.turn.captured0);
  CHECK(s.game.turn.captured1 == before.game.turn.captured1 && s.bombs0 == 0 && s.credits0 == 0);
  return true;
}

bool arithmetic_and_illegal_case() {
  auto s = base();
  s.shaken0 = static_cast<std::uint16_t>((1u << 0) | (1u << 4));
  s.bombs0 = 2;
  CHECK(shake_count50(s, 0) == 2 && bomb_shake_multiplier50(s, 0) == 16);
  auto bad = base();
  bad.game.turn.hand0 = card_bit(0) | card_bit(1);
  bad.game.turn.floor = card_bit(2);
  bad.game.turn.stock = card_bit(8);
  const auto before = bad;
  CHECK(play_bomb50(bad, 0).status == SpecialStatus50::kNotLegal);
  CHECK(bad.game.turn.hand0 == before.game.turn.hand0 && bad.game.turn.floor == before.game.turn.floor);
  return true;
}
}  // namespace

int main() {
  if (!shake_case() || !bomb_case() || !grenade_case() || !credit_case() ||
      !rollback_case() || !arithmetic_and_illegal_case()) return 1;
  std::cout << "cugo_special50_test: PASS\n";
  return 0;
}
