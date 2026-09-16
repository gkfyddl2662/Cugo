#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "cugo/game/postchance50.h"

namespace {

using namespace cugo::game;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_full_rollout_equivalence(std::uint64_t& resolve_choices,
                                   std::uint64_t& pi_choices) {
  constexpr std::uint32_t kSamples = 8192;
  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const std::uint64_t seed =
        0xbb67ae8584caa73bull +
        static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull;
    const std::uint8_t first = static_cast<std::uint8_t>(i & 1u);

    const RolloutDigest50 baseline =
        rollout_digest50(rollout_canonical50(seed, first));
    TerminalGameState50 state =
        make_terminal_game_state50(deal_shin_matgo_50(seed), first);
    const PostChanceRollout50 explicit_result =
        rollout_canonical_postchance50(state);
    const RolloutDigest50 explicit_digest =
        rollout_digest50(explicit_result.result);

    require(equal_rollout_digest50(baseline, explicit_digest),
            "explicit post-chance rollout must match canonical rollout");
    require(explicit_result.result.end == RolloutEnd50::kTerminal ||
                explicit_result.result.end == RolloutEnd50::kNagari,
            "explicit post-chance rollout must finish");
    require(explicit_result.result.action_status == ActionStatus50::kOk,
            "explicit post-chance rollout action status");
    require(is_valid_terminal_game_state50(state),
            "explicit post-chance final state must be valid");

    resolve_choices += explicit_result.resolve_choices;
    pi_choices += explicit_result.pi_choices;
  }
}

void test_packet_contract() {
  constexpr std::uint32_t kSearch = 16384;
  bool found = false;

  for (std::uint32_t i = 0; i < kSearch && !found; ++i) {
    const std::uint64_t seed =
        0x3c6ef372fe94f82bull +
        static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull;
    TerminalGameState50 state = make_terminal_game_state50(
        deal_shin_matgo_50(seed), static_cast<std::uint8_t>(i & 1u));

    for (std::uint16_t step_index = 0; step_index < 64 && !found; ++step_index) {
      if (terminal50_is_finished(state) || rollout_hands_exhausted50(state)) break;
      const CanonicalAction50 canonical = canonical_action50(state);
      require(canonical.status == ActionStatus50::kOk,
              "canonical action while searching packet");
      const std::uint16_t primary = primary_action_index50(canonical.action);
      require(primary != kInvalidPolicyAction50, "primary action index");

      PostChancePending50 pending{};
      PostChanceStep50 step = start_postchance_action50(state, primary, pending);
      require(step.status == ActionStatus50::kOk,
              "start post-chance action while searching packet");
      if (!step.action_committed) {
        found = true;
        require(step.packet.legal_cards != 0, "post-chance legal cards");
        require(step.packet.decision != PostChanceDecision50::kNone,
                "post-chance decision kind");
        require(step.packet.decision_player <= 1u,
                "post-chance decision player");
        const TurnState50& turn = state.special.game.turn;
        const CardMask opponent_hand = step.packet.decision_player == 0
            ? turn.hand1 : turn.hand0;
        require((opponent_hand & ~step.packet.observation.unseen) == 0,
                "post-chance observation must not reveal opponent hand");

        const CardId choice = core::first_card(step.packet.legal_cards);
        const TerminalGameState50 before = state;
        const CardId bad = core::kInvalidCard;
        const PostChanceStep50 rejected =
            choose_postchance_card50(state, pending, bad);
        require(rejected.status == ActionStatus50::kInvalidResolveChoice ||
                    rejected.status == ActionStatus50::kInvalidPiSelection,
                "invalid post-chance choice must be rejected");
        require(state.special.game.turn.hand0 == before.special.game.turn.hand0 &&
                    state.special.game.turn.hand1 == before.special.game.turn.hand1 &&
                    state.special.game.turn.stock == before.special.game.turn.stock &&
                    state.special.game.turn.floor == before.special.game.turn.floor,
                "invalid post-chance choice must not mutate base state");

        step = choose_postchance_card50(state, pending, choice);
        require(step.status == ActionStatus50::kOk,
                "valid post-chance choice");
      }

      while (step.status == ActionStatus50::kOk && !step.action_committed) {
        const CardId choice = core::first_card(step.packet.legal_cards);
        step = choose_postchance_card50(state, pending, choice);
      }
      require(step.status == ActionStatus50::kOk && step.action_committed,
              "post-chance action must eventually commit");
    }
  }

  require(found, "must find at least one post-chance decision packet");
}

}  // namespace

int main() {
  try {
    std::uint64_t resolve_choices = 0;
    std::uint64_t pi_choices = 0;
    test_full_rollout_equivalence(resolve_choices, pi_choices);
    test_packet_contract();
    require(resolve_choices != 0, "must exercise resolve choices");
    require(pi_choices != 0, "must exercise pi choices");
    std::cout << "cugo_postchance50_test: PASS (8192 explicit rollouts; resolve_choices="
              << resolve_choices << " pi_choices=" << pi_choices
              << " packet_bytes=" << sizeof(PostChancePacket50) << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cugo_postchance50_test: FAIL: " << e.what() << '\n';
    return 1;
  }
}
