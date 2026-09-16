#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "cugo/game/rollout50.h"

namespace {

using namespace cugo::game;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_determinism() {
  constexpr std::uint64_t seed = 0x123456789abcdef0ull;
  const RolloutDigest50 a = rollout_digest50(rollout_canonical50(seed, 0));
  const RolloutDigest50 b = rollout_digest50(rollout_canonical50(seed, 0));
  require(equal_rollout_digest50(a, b), "rollout must be deterministic");
}

void test_full_games() {
  constexpr std::uint32_t kSamples = 4096;
  std::uint32_t terminal = 0;
  std::uint32_t nagari = 0;
  std::uint64_t bonus_actions = 0;
  std::uint64_t special_actions = 0;
  std::uint64_t go_actions = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const std::uint64_t seed =
        0x6a09e667f3bcc909ull +
        static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull;
    TerminalGameState50 state = make_terminal_game_state50(
        deal_shin_matgo_50(seed), static_cast<std::uint8_t>(i & 1u));
    const RolloutResult50 result = rollout_canonical50(state);

    require(result.end == RolloutEnd50::kTerminal ||
                result.end == RolloutEnd50::kNagari,
            "full rollout must finish or nagari");
    require(result.action_status == ActionStatus50::kOk,
            "completed rollout must not carry an action error");
    require(result.actions <= 128, "action cap");
    require(result.turns <= 64, "turn count sanity");

    if (result.end == RolloutEnd50::kTerminal) {
      ++terminal;
      require(terminal50_is_finished(state), "terminal rollout state");
      const Settlement50 settlement = settle_terminal50(state);
      require(settlement.status == SettlementStatus50::kOk,
              "terminal rollout must settle");
      require(result.final_points == settlement.final_points,
              "rollout settlement points");
      require(result.reward0 == settlement_reward50(settlement, 0),
              "rollout reward");
      require(result.next_round_multiplier == 1,
              "terminal next-round multiplier");
    } else {
      ++nagari;
      require(!terminal50_is_finished(state), "nagari is not a winner terminal");
      require(rollout_hands_exhausted50(state), "nagari must exhaust hands");
      require(result.final_points == 0 && result.reward0 == 0,
              "nagari has zero terminal reward");
      require(result.next_round_multiplier == 2,
              "nagari doubles the next scored round");
    }

    bonus_actions += result.bonus_actions;
    special_actions += result.special_actions;
    go_actions += result.go_actions;
  }

  require(terminal + nagari == kSamples, "all samples accounted for");
  require(bonus_actions != 0, "rollout must exercise bonus actions");
  require(special_actions != 0, "rollout must exercise special actions");
  require(go_actions != 0, "rollout must exercise Go decisions");
}

}  // namespace

int main() {
  try {
    test_determinism();
    test_full_games();
    std::cout << "cugo_rollout50_test: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cugo_rollout50_test: FAIL: " << e.what() << '\n';
    return 1;
  }
}
