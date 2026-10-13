// Hangame Shin Matgo rule differences (SingleServer), one check per rule.
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "cugo/game/policy50.h"
#include "cugo/game/settlement50.h"
#include "cugo/game/torch50.h"

namespace {

using cugo::core::CardId;
using cugo::core::CardMask;
using cugo::core::card_bit;
using cugo::core::card_count;
using cugo::core::make_card;
using namespace cugo::game;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

CardMask pi_cards(std::uint8_t first_month, int count) {
  CardMask out = 0;
  for (std::uint8_t month = first_month; count > 0; ++month) {
    for (std::uint8_t slot = 2; slot < 4 && count > 0; ++slot, --count) {
      out |= card_bit(make_card(month, slot));
    }
  }
  return out;
}

// Resolve-phase turn with every card not otherwise placed left in the stock.
TurnState50 resolve_state(CardMask floor, CardId played, CardId drawn,
                          CardMask captured0, CardMask captured1) {
  TurnState50 state{};
  state.actor = 0;
  state.phase = Turn50Phase::kResolve;
  state.floor = floor;
  state.captured0 = captured0;
  state.captured1 = captured1;
  state.pending_played = played;
  state.pending_drawn = drawn;
  state.stock = cugo::core::kShinMatgoDeckMask &
                ~(floor | captured0 | captured1 | card_bit(played) | card_bit(drawn));
  return state;
}

TerminalGameState50 terminal_from_turn(const TurnState50& turn) {
  TerminalGameState50 state{};
  state.special.game.turn = turn;
  state.special.game.decision_actor = kNoGame50Player;
  state.special.game.winner = kNoGame50Player;
  state.pending_chongtong_actor = kNoTerminal50Player;
  state.terminal_winner = kNoTerminal50Player;
  state.round_multiplier = 1;
  return state;
}

// 1. Initial floor bonuses go to the first player and the floor is refilled.
void test_initial_floor_refill() {
  int refilled = 0;
  int triples = 0;
  for (std::uint64_t seed = 0; seed < 20000; ++seed) {
    const InitialDeal50 deal = deal_shin_matgo_50(seed);
    const std::uint8_t first = static_cast<std::uint8_t>(seed & 1u);
    const TurnState50 state = make_turn_state50(deal, first);
    const CardMask floor_bonus = deal.floor & cugo::core::kBonusCardMask;
    const CardMask first_captured = first == 0 ? state.captured0 : state.captured1;
    const CardMask other_captured = first == 0 ? state.captured1 : state.captured0;
    require((state.floor & cugo::core::kBonusCardMask) == 0, "refilled floor has no bonus");
    require(card_count(state.floor) == card_count(deal.floor), "floor size is kept");
    require((first_captured & floor_bonus) == floor_bonus, "floor bonus goes to first player");
    require(other_captured == 0, "second player gets nothing");
    const CardMask all = state.hand0 | state.hand1 | state.floor | state.stock |
                         state.captured0 | state.captured1;
    require(all == cugo::core::kShinMatgoDeckMask, "refill conserves cards");
    require(card_count(state.hand0) + card_count(state.hand1) + card_count(state.floor) +
                    card_count(state.stock) + card_count(state.captured0) +
                    card_count(state.captured1) ==
                card_count(cugo::core::kShinMatgoDeckMask),
            "refill keeps piles disjoint");
    require(state.neutral_ppuk_months == floor_triple_months50(state.floor),
            "dealt triples become neutral ppuk");
    require(is_valid_turn_state50(state), "refilled state is valid");
    refilled += floor_bonus != 0;
    triples += state.neutral_ppuk_months != 0;
  }
  require(refilled != 0, "must exercise floor refills");
  require(triples != 0, "must exercise neutral ppuk deals");
}

// 2. Capturing a dealt triple steals one pi.
void test_neutral_ppuk_capture() {
  const CardMask triple = card_bit(make_card(3, 0)) | card_bit(make_card(3, 1)) |
                          card_bit(make_card(3, 2));
  TurnState50 state = resolve_state(triple | card_bit(make_card(9, 0)), make_card(3, 3),
                                    make_card(5, 0), 0, pi_cards(6, 2));
  state.neutral_ppuk_months = 1u << 3;
  require(is_valid_turn_state50(state), "neutral ppuk fixture is valid");
  const auto result = resolve_turn50_with_pi_transfer(state);
  require(result.resolve.status == Resolve50Status::kOk, "neutral ppuk resolves");
  require(result.resolve.captured_opponent_ppuk == 1, "neutral ppuk counts as a stack");
  require(result.pi_transfer.transferred_cards == 1, "neutral ppuk steals one pi");
  require(state.neutral_ppuk_months == 0, "neutral ppuk is consumed");
  require((state.captured0 & (triple | card_bit(make_card(3, 3)))) ==
              (triple | card_bit(make_card(3, 3))),
          "all four cards captured");
}

// 3. Turns 19 and 20: no ppuk, no pi steals, automatic stop.
void test_last_turn_rules() {
  Resolve50Result counted{};
  counted.status = Resolve50Status::kOk;
  counted.events = kResolve50EventJjok | kResolve50EventSweep;
  counted.captured_own_ppuk = 1;
  counted.captured_opponent_ppuk = 1;
  counted.captured_cards = card_bit(cugo::core::kBonusTwoPi);
  counted.last_turn = 1;
  require(resolve_pi_steal_card_count(counted) == 0, "last turn steals nothing");
  counted.last_turn = 0;
  require(resolve_pi_steal_card_count(counted) == 1 + 2 + 1 + 1 + 1,
          "jjok, own stack, opponent stack, sweep and a bonus card each steal");

  // Jjok on the last turn takes the cards but steals nothing.
  const CardMask victim = pi_cards(6, 2);
  TurnState50 jjok = resolve_state(card_bit(make_card(1, 0)), make_card(0, 0),
                                   make_card(0, 1), 0, victim);
  jjok.turn_index = kLastTurnIndex50;
  auto result = resolve_turn50_with_pi_transfer(jjok);
  require(result.resolve.status == Resolve50Status::kOk, "last jjok resolves");
  require((result.resolve.events & kResolve50EventJjok) == 0, "no jjok event on last turn");
  require(result.pi_transfer.transferred_cards == 0, "no steal on last turn");
  require(jjok.captured1 == victim, "victim keeps pi");

  // A would-be ppuk on the last turn is captured instead.
  TurnState50 ppuk = resolve_state(card_bit(make_card(1, 0)), make_card(1, 1),
                                   make_card(1, 2), 0, victim);
  ppuk.turn_index = kLastTurnIndex50;
  result = resolve_turn50_with_pi_transfer(ppuk);
  require(result.resolve.status == Resolve50Status::kOk, "last ppuk resolves");
  require((result.resolve.events & kResolve50EventPpuk) == 0, "no ppuk on last turn");
  require(ppuk.ppuk_months == 0, "no ppuk stack on last turn");
  require((ppuk.captured0 & card_bit(make_card(1, 0))) != 0, "matched card captured");

  // The same play before the last turn makes a ppuk.
  TurnState50 early = resolve_state(card_bit(make_card(1, 0)), make_card(1, 1),
                                    make_card(1, 2), 0, victim);
  early.turn_index = kLastTurnIndex50 - 1;
  result = resolve_turn50_with_pi_transfer(early);
  require((result.resolve.events & kResolve50EventPpuk) != 0, "ppuk before last turn");

  // A hand bonus on the last turn steals nothing.
  TurnState50 bonus{};
  bonus.actor = 0;
  bonus.phase = Turn50Phase::kPlay;
  bonus.hand0 = card_bit(cugo::core::kBonusTwoPi);
  bonus.stock = card_bit(make_card(0, 0));
  bonus.captured1 = card_bit(make_card(1, 2));
  bonus.turn_index = kLastTurnIndex50;
  const auto bonus_result =
      play_bonus_for_turn50_with_pi_transfer(bonus, cugo::core::kBonusTwoPi);
  require(bonus_result.play.status == Turn50Status::kOk, "last bonus plays");
  require(bonus_result.pi_transfer.transferred_cards == 0, "last bonus steals nothing");

  // Reaching the Go/Stop score after turn 19 stops automatically.
  GameState50 game{};
  game.turn.captured0 = card_bit(make_card(0, 0)) | card_bit(make_card(2, 0)) |
                        card_bit(make_card(7, 0)) | kGodoriMask;
  game.decision_actor = kNoGame50Player;
  game.winner = kNoGame50Player;
  require(game50_base_score(game, 0) >= kShinMatgoDecisionBasePoints, "score reaches stop");
  GameState50 mid = game;
  mid.turn.turn_index = 10;
  require(maybe_open_go_stop_decision50(mid, 0), "mid game opens Go/Stop");
  require(mid.decision_actor == 0 && mid.winner == kNoGame50Player, "mid game asks");
  GameState50 late = game;
  late.turn.turn_index = kLastTurnIndex50 + 1;
  require(!maybe_open_go_stop_decision50(late, 0), "last turns do not ask");
  require(late.winner == 0 && late.decision_actor == kNoGame50Player, "last turns auto-stop");
}

// 4. Pi-bak needs 1..7 loser pi; a loser who took nothing makes the game void.
void test_settlement_rules() {
  // Winner: three brights (3) + godori (5) + ten pi (1) = 9 points.
  const CardMask winner = card_bit(make_card(0, 0)) | card_bit(make_card(2, 0)) |
                          card_bit(make_card(7, 0)) | kGodoriMask | pi_cards(0, 10);
  const CardMask ribbon = card_bit(make_card(9, 1));
  auto settle = [&](CardMask loser, std::uint8_t round, std::uint8_t mission) {
    TurnState50 turn{};
    turn.captured0 = winner;
    turn.captured1 = loser;
    TerminalGameState50 state = terminal_from_turn(turn);
    state.round_multiplier = round;
    state.mission_multi0 = mission;
    state.terminal_winner = 0;
    state.terminal_reason = TerminalReason50::kStop;
    return settle_terminal50(state);
  };

  Settlement50 s = settle(0, 1, 0);
  require(s.status == SettlementStatus50::kOk, "void settles");
  require(s.winner_base_points == 9, "fixture scores 9");
  require((s.flags & kSettlementVoid) != 0 && s.final_points == 0, "void game pays 0");

  s = settle(ribbon, 1, 0);
  require((s.flags & kSettlementPiBak) == 0, "zero loser pi is exempt from pi-bak");
  require(s.final_points == 9u * 2u, "gwang-bak only");

  s = settle(ribbon | pi_cards(5, 5), 1, 0);
  require((s.flags & kSettlementPiBak) != 0, "five loser pi is pi-bak");
  require(s.final_points == 9u * 4u, "gwang-bak and pi-bak");

  s = settle(ribbon | pi_cards(5, 8), 1, 0);
  require((s.flags & kSettlementPiBak) == 0, "eight loser pi escapes pi-bak");

  s = settle(ribbon | pi_cards(5, 5), 2, 3);
  require(s.round_multiplier == 2 && s.mission_multiplier == 3, "carry and mission recorded");
  require(s.final_points == 9u * 4u * 2u * 3u, "carry and mission multiply");

  // Event payments are scaled by the carry and count even on nagari.
  TerminalGameState50 paid = terminal_from_turn(TurnState50{});
  paid.round_multiplier = 2;
  add_side_payment50(paid, 1, kEventPayment50);
  require(paid.side_points0 == -14, "side payment scaled by carry");
  require(nagari_reward50(paid, 1) == 14 && nagari_reward50(paid, 0) == -14,
          "nagari keeps side payments");
}

// 5. Gukjin: animal by default, one prompt, loser auto-decision, lock on shortfall.
void test_gukjin_rules() {
  const CardMask brights = card_bit(make_card(0, 0)) | card_bit(make_card(2, 0)) |
                           card_bit(make_card(7, 0));
  TurnState50 turn{};
  turn.captured0 = card_bit(kGukjin) | brights | pi_cards(0, 8);
  require(gukjin_undecided50(turn, 0), "gukjin starts undecided");
  require(persistent_gukjin_role50(turn, 0) == PersistentGukjinRole50::kAnimal,
          "gukjin defaults to animal");
  // As pi: 3 brights (3) + 10 pi (1) = 4; the go count tops it up to 7.
  require(!gukjin_prompt_due50(turn, 0, 2), "4 + 2 go is short of 7");
  require(gukjin_prompt_due50(turn, 0, 3), "4 + 3 go reaches 7");
  TurnState50 few = turn;
  few.captured0 &= ~card_bit(make_card(0, 2));
  require(!gukjin_prompt_due50(few, 0, 6), "needs eight pi before converting");

  // The prompt comes before Go/Stop and is asked once.
  GameState50 game{};
  game.turn = turn;
  game.turn.captured0 |= kGodoriMask & ~card_bit(kGukjin);
  game.turn.turn_index = 6;
  game.decision_actor = kNoGame50Player;
  game.winner = kNoGame50Player;
  require(finish_turn_game50(game, 0), "prompt opens");
  require(game50_has_gukjin_decision(game) && game50_gukjin_actor(game) == 0,
          "gukjin prompt for the actor");
  require(game.decision_actor == kNoGame50Player, "Go/Stop waits for the prompt");
  require(apply_gukjin_decision50(game, true) == GoStopStatus50::kOk, "answer prompt");
  require(gukjin_locked50(game.turn, 0), "answer locks gukjin");
  require(persistent_gukjin_role50(game.turn, 0) == PersistentGukjinRole50::kDoublePi,
          "answered as double pi");
  require(game.decision_actor == 0, "Go/Stop follows the prompt");
  require(!gukjin_prompt_due50(game.turn, 0, 9), "prompt is one-time");

  // Loser auto-decision: 6-7 pi converts, 0 pi stays animal, otherwise undecided.
  auto loser_with = [](int pi) {
    TurnState50 t{};
    t.captured1 = card_bit(kGukjin) | pi_cards(0, pi);
    auto_decide_loser_gukjin50(t, 1);
    return t;
  };
  TurnState50 t = loser_with(6);
  require(persistent_gukjin_role50(t, 1) == PersistentGukjinRole50::kDoublePi,
          "6 pi + gukjin as pi escapes pi-bak");
  t = loser_with(0);
  require(gukjin_locked50(t, 1) &&
              persistent_gukjin_role50(t, 1) == PersistentGukjinRole50::kAnimal,
          "zero pi locks animal");
  t = loser_with(7);
  require(persistent_gukjin_role50(t, 1) == PersistentGukjinRole50::kDoublePi,
          "7 pi converts too");
  t = loser_with(5);
  require(gukjin_undecided50(t, 1), "other counts stay undecided");

  // A steal that finds no pi locks the victim's gukjin as an animal.
  TurnState50 victim{};
  victim.captured1 = card_bit(kGukjin);
  const auto steal = apply_pi_steal50(victim, 0, 1);
  require(steal.status == PiTransferStatus::kOk && steal.transferred_cards == 0,
          "nothing to steal");
  require(gukjin_locked50(victim, 1), "shortfall locks the victim's gukjin");
}

// 6. Bomb and grenade are forced; a whole month in hand offers a four-card shake.
void test_forced_specials_and_four_shake() {
  auto special_state = [](CardMask hand, CardMask floor) {
    TurnState50 turn{};
    turn.actor = 0;
    turn.phase = Turn50Phase::kPlay;
    turn.hand0 = hand;
    turn.floor = floor;
    turn.pending_played = cugo::core::kInvalidCard;
    turn.pending_drawn = cugo::core::kInvalidCard;
    turn.stock = cugo::core::kShinMatgoDeckMask & ~(hand | floor);
    return terminal_from_turn(turn);
  };
  const CardId other = make_card(6, 0);

  TerminalGameState50 bomb = special_state(
      card_bit(make_card(4, 0)) | card_bit(make_card(4, 1)) | card_bit(make_card(4, 2)) |
          card_bit(other),
      card_bit(make_card(4, 3)) | card_bit(make_card(9, 0)));
  require(forced_special_months50(bomb.special) == (1u << 4), "bomb month is forced");
  PolicyActionMask50 mask = legal_primary_actions50(bomb);
  require(!has_policy_action50(mask, make_card(4, 0)), "forced month cannot be played");
  require(has_policy_action50(mask, other), "other cards stay playable");
  require(has_policy_action50(mask, kPolicyBombBegin50 + 4), "bomb is offered");

  TerminalGameState50 grenade = special_state(
      card_bit(make_card(4, 0)) | card_bit(make_card(4, 1)) | card_bit(other),
      card_bit(make_card(4, 2)) | card_bit(make_card(4, 3)));
  require(forced_special_months50(grenade.special) == (1u << 4), "grenade month is forced");
  mask = legal_primary_actions50(grenade);
  require(!has_policy_action50(mask, make_card(4, 1)), "forced grenade month");
  require(has_policy_action50(mask, kPolicyGrenadeBegin50 + 4), "grenade is offered");

  TerminalGameState50 four = special_state(
      cugo::core::month_mask(4) | card_bit(other), card_bit(make_card(9, 0)));
  require((legal_shake_months50(four.special) & (1u << 4)) != 0, "four-card shake offered");
  require(begin_shake50(four.special, make_card(4, 0)).status == SpecialStatus50::kOk,
          "four-card shake plays");
  require(shaken4_50(four.special, 0) == (1u << 4) && shaken50(four.special, 0) == 0,
          "recorded as a four-card shake");
  require(shake_count50(four.special, 0) == 1 &&
              bomb_shake_multiplier50(four.special, 0) == 2,
          "four-card shake doubles");
}

// 7. Missions: completion, triple, and the draw distribution.
void test_missions() {
  TerminalGameState50 state = terminal_from_turn(TurnState50{});
  state.mission_type = kMissionEasy50;
  state.mission_multiplier = 3;
  state.special.game.turn.mission_cards = hg_cards50(1, 2);
  state.special.game.turn.captured0 = card_bit(0);
  update_missions50(state, 0, card_bit(0));
  require(mission_multi50(state, 0) == 0, "half a mission pays nothing");
  state.special.game.turn.captured0 |= card_bit(1);
  update_missions50(state, 0, card_bit(1));
  require(mission_multi50(state, 0) == 3, "completed mission pays its multiplier");
  require(mission_multi50(state, 1) == 0, "opponent unaffected");

  state.mission_type = kMissionTriple50;
  state.mission_multiplier = 0;
  state.special.game.turn.mission_cards = mission_cards50(kMissionTriple50, 0);
  state.special.game.turn.captured1 =
      state.special.game.turn.mission_cards & ~card_bit(cugo::core::first_card(
                                                  state.special.game.turn.mission_cards));
  update_missions50(state, 1, 0);
  require(mission_multi50(state, 1) == 4, "triple pays 2^owned");

  int seen[7] = {};
  for (std::uint64_t seed = 0; seed < 50000; ++seed) {
    TerminalGameState50 m = terminal_from_turn(TurnState50{});
    assign_mission50(m, seed);
    require(m.mission_type <= kMissionTriple50, "valid mission type");
    ++seen[m.mission_type];
    if (m.mission_type == kMissionNoCondition50) {
      require(m.mission_multi0 == m.mission_multi1 && m.mission_multi0 >= 2 &&
                  m.mission_multi0 <= 4,
              "no-condition pays both players");
      require(m.special.game.turn.mission_cards == 0, "no-condition has no cards");
    } else if (m.mission_type == kMissionTriple50) {
      require(card_count(m.special.game.turn.mission_cards) == 3, "triple has three cards");
    } else {
      require(m.mission_multiplier >= 2 && m.mission_multiplier <= 4, "multiplier 2..4");
      require(m.special.game.turn.mission_cards != 0, "mission has cards");
    }
  }
  for (std::uint8_t type : {kMissionEasy50, kMissionNormal50, kMissionHard50,
                            kMissionNoCondition50, kMissionTriple50}) {
    require(seen[type] > 9000 && seen[type] < 11000, "mission types are uniform");
  }
}

// 8. Steal order, automatic floor choice, sweep, and first ttadak.
void test_steal_order_choice_sweep_ttadak() {
  const CardId mission_pi = make_card(0, 2);
  const CardId plain_pi = make_card(1, 2);
  const CardId double_pi = make_card(10, 1);
  const CardMask candidates = card_bit(mission_pi) | card_bit(plain_pi) | card_bit(double_pi);
  const CardMask mission = card_bit(mission_pi);
  require(hangame_pi_steal_selection50(candidates, mission, false, 1) == card_bit(plain_pi),
          "non-mission first on equal value");
  require(hangame_pi_steal_selection50(candidates, mission, false, 2) ==
              (card_bit(plain_pi) | card_bit(mission_pi)),
          "cheapest pi before double pi");
  require(hangame_pi_steal_selection50(card_bit(mission_pi) | card_bit(double_pi), mission,
                                       true, 1) == card_bit(double_pi),
          "mission owner loses non-mission cards first");
  require(hangame_pi_steal_selection50(card_bit(mission_pi) | card_bit(double_pi), mission,
                                       false, 1) == card_bit(mission_pi),
          "otherwise value comes first");

  require(auto_floor_choice50(card_bit(make_card(0, 2)) | card_bit(make_card(0, 3))),
          "two plain pi are interchangeable");
  require(!auto_floor_choice50(card_bit(make_card(0, 1)) | card_bit(make_card(0, 2))),
          "ribbon vs pi is a real choice");
  require(!auto_floor_choice50(card_bit(make_card(11, 2)) | card_bit(make_card(11, 3))),
          "December is a real choice");

  TurnState50 sweep = resolve_state(card_bit(make_card(2, 0)) | card_bit(make_card(5, 0)),
                                    make_card(2, 1), make_card(5, 1), 0, pi_cards(6, 2));
  auto result = resolve_turn50_with_pi_transfer(sweep);
  require((result.resolve.events & kResolve50EventSweep) != 0, "empty floor is a sweep");
  require(result.pi_transfer.transferred_cards == 1, "sweep steals one");

  auto ttadak_at = [](std::uint16_t turn_index) {
    TurnState50 turn = resolve_state(card_bit(make_card(2, 0)) | card_bit(make_card(2, 1)) |
                                         card_bit(make_card(9, 0)),
                                     make_card(2, 2), make_card(2, 3), 0, pi_cards(6, 2));
    turn.turn_index = turn_index;
    TerminalGameState50 state = terminal_from_turn(turn);
    const auto r = resolve_terminal_turn50_with_pi_transfer(state);
    require(r.game_result.turn_result.resolve.status == Resolve50Status::kOk, "ttadak resolves");
    require((r.game_result.turn_result.resolve.events & kResolve50EventTtadak) != 0,
            "ttadak event");
    return state.side_points0;
  };
  require(ttadak_at(0) == kEventPayment50, "first-turn ttadak pays 7");
  require(ttadak_at(2) == 0, "later ttadak pays nothing extra");
}

}  // namespace

// 9. Public information in the observation: neutral ppuk, four-card shakes, cards a
// shake revealed, and each side's played cards.
void test_public_observation() {
  // Neutral ppuk has its own plane and belongs to neither player.
  TurnState50 neutral{};
  neutral.actor = 0;
  neutral.phase = Turn50Phase::kPlay;
  neutral.pending_played = cugo::core::kInvalidCard;
  neutral.pending_drawn = cugo::core::kInvalidCard;
  neutral.floor = card_bit(make_card(3, 0)) | card_bit(make_card(3, 1)) |
                  card_bit(make_card(3, 2));
  neutral.hand0 = card_bit(make_card(6, 0));
  neutral.hand1 = card_bit(make_card(7, 0));
  neutral.stock = cugo::core::kShinMatgoDeckMask & ~(neutral.floor | neutral.hand0 | neutral.hand1);
  neutral.neutral_ppuk_months = 1u << 3;
  const PolicyObservation50 n = pack_policy_observation50(terminal_from_turn(neutral));
  require(n.neutral_ppuk_months == (1u << 3), "neutral ppuk is observed");
  require(((n.own_ppuk_months | n.opponent_ppuk_months | n.ppuk_months) & (1u << 3)) == 0,
          "neutral ppuk has no owner");

  // A shake reveals the month's cards; the opponent sees the ones still in hand.
  TurnState50 turn{};
  turn.actor = 0;
  turn.phase = Turn50Phase::kPlay;
  turn.pending_played = cugo::core::kInvalidCard;
  turn.pending_drawn = cugo::core::kInvalidCard;
  turn.hand0 = card_bit(make_card(4, 0)) | card_bit(make_card(4, 1)) |
               card_bit(make_card(4, 2)) | card_bit(make_card(6, 0));
  turn.hand1 = card_bit(make_card(7, 0)) | card_bit(make_card(8, 0));
  turn.floor = card_bit(make_card(9, 0)) | card_bit(make_card(10, 0));
  turn.stock = cugo::core::kShinMatgoDeckMask & ~(turn.hand0 | turn.hand1 | turn.floor);
  turn.rng_state = 7;
  TerminalGameState50 game = terminal_from_turn(turn);
  Action50 shake = make_action50(ActionKind50::kShake);
  shake.card = make_card(4, 0);
  shake.month = 4;
  require(apply_action50(game, shake).status == ActionStatus50::kOk, "shake applies");
  require(game.revealed0 == (cugo::core::month_mask(4) & turn.hand0), "shake reveals three cards");
  require(game.special.game.turn.actor == 1, "turn passes");
  const PolicyObservation50 seen = pack_policy_observation50(game);
  const CardMask still = card_bit(make_card(4, 1)) | card_bit(make_card(4, 2));
  require(seen.opponent_known_hand == still, "revealed cards still in hand are known");
  require((seen.unseen & still) == 0, "known cards leave the unseen plane");
  require((seen.opponent_shaken_months & (1u << 4)) != 0, "opponent shake is observed");
  require(seen.opponent_played == card_bit(make_card(4, 0)), "opponent's played card");
  require(seen.own_played == 0, "nothing played yet by the observer");

  // Four-card shakes show up in the shake planes like three-card ones.
  TerminalGameState50 four = terminal_from_turn(turn);
  four.special.shaken4_1 = 1u << 5;
  require((pack_policy_observation50(four).opponent_shaken_months & (1u << 5)) != 0,
          "four-card shake is observed");
}

// 10. Hangame records a month once its shake question is answered and never asks
// again. Cugo keeps no such record; random games check that a declined month (a
// card of a shakeable month played without shaking) can never be offered again.
void test_declined_shake_never_returns() {
  std::uint64_t rng = 0x5eedf00dULL;
  auto next = [&rng]() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(rng >> 33);
  };
  int declines = 0;
  for (int g = 0; g < 4000; ++g) {
    TorchEnv50 env = make_torch_env50(0x9e3779b97f4a7c15ULL * (g + 1), static_cast<std::uint8_t>(g & 1));
    std::uint16_t declined[2] = {0, 0};
    for (int step = 0; step < 200 && env.done == 0; ++step) {
      const TorchPacket50 packet = make_torch_packet50(env);
      require(packet.status == ActionStatus50::kOk, "packet");
      const std::uint8_t actor = env.game.special.game.turn.actor;
      const bool primary = env.pending.active == 0 &&
          packet.decision_kind == static_cast<std::uint8_t>(PolicyDecision50::kPrimary);
      std::uint16_t shakeable = 0;
      if (primary) {
        shakeable = legal_shake_months50(env.game.special);
        require((shakeable & declined[actor]) == 0, "a declined shake is never offered again");
      }
      const CardMask hand_before = actor == 0 ? env.game.special.game.turn.hand0
                                              : env.game.special.game.turn.hand1;
      const std::uint16_t shaken_before = static_cast<std::uint16_t>(
          shaken50(env.game.special, actor) | shaken4_50(env.game.special, actor));

      std::uint16_t legal[kTorch50ActionCount];
      int count = 0;
      for (std::uint16_t a = 0; a < kTorch50ActionCount; ++a)
        if (has_torch_action50(packet.legal_actions, a)) legal[count++] = a;
      require(count > 0, "legal action exists");
      require(torch_step50(env, legal[next() % count]).status == ActionStatus50::kOk, "step");

      if (shakeable == 0) continue;
      const CardMask hand_after = actor == 0 ? env.game.special.game.turn.hand0
                                             : env.game.special.game.turn.hand1;
      const std::uint16_t shaken_after = static_cast<std::uint16_t>(
          shaken50(env.game.special, actor) | shaken4_50(env.game.special, actor));
      for (std::uint8_t m = 0; m < cugo::core::kMonthCount; ++m) {
        const std::uint16_t bit = static_cast<std::uint16_t>(1u << m);
        const bool played_month =
            ((hand_before & ~hand_after) & cugo::core::month_mask(m)) != 0;
        if ((shakeable & bit) != 0 && played_month && (shaken_after & ~shaken_before & bit) == 0) {
          declined[actor] |= bit;
          ++declines;
        }
      }
    }
  }
  require(declines > 100, "random games decline shakes often enough to test");
}

int main() {
  try {
    test_initial_floor_refill();
    test_neutral_ppuk_capture();
    test_last_turn_rules();
    test_settlement_rules();
    test_gukjin_rules();
    test_forced_specials_and_four_shake();
    test_missions();
    test_steal_order_choice_sweep_ttadak();
    test_public_observation();
    test_declined_shake_never_returns();
  } catch (const std::exception& error) {
    std::cerr << "cugo_hangame_rules50_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "cugo_hangame_rules50_test: PASS\n";
  return 0;
}
