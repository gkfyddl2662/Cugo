#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "cugo/game/torch50.h"

namespace {

using namespace cugo::game;
namespace core = cugo::core;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void verify_packet_contract(const TorchEnv50& env) {
  const TorchPacket50 packet = make_torch_packet50(env);
  require(packet.status == ActionStatus50::kOk, "torch packet status");

  if (env.done != 0) {
    require(torch_action_count50(packet.legal_actions) == 0,
            "finished env must have no legal actions");
    return;
  }

  if (env.pending.active == 0) {
    const PolicyActionMask50 expected = legal_primary_actions50(env.game);
    for (std::uint16_t action = 0; action < kPolicyActionCount50; ++action) {
      require(has_torch_action50(packet.legal_actions, action) ==
                  has_policy_action50(expected, action),
              "primary legal mask mismatch");
    }
    for (std::uint16_t card = 0; card < core::kShinMatgoCardCount; ++card) {
      require(!has_torch_action50(
                  packet.legal_actions,
                  static_cast<std::uint16_t>(kTorch50ChoiceActionBegin + card)),
              "primary packet must not expose choice actions");
    }
  } else {
    TerminalGameState50 staged{};
    const PostChanceProbe50 probe = probe_postchance50(env.game, env.pending, staged);
    require(probe.status == ActionStatus50::kOk, "post-chance probe status");
    require(probe.decision != PostChanceDecision50::kNone,
            "pending env must expose a post-chance decision");
    for (std::uint16_t action = 0; action < kPolicyActionCount50; ++action) {
      require(!has_torch_action50(packet.legal_actions, action),
              "post-chance packet must hide primary actions");
    }
    for (std::uint16_t card = 0; card < core::kShinMatgoCardCount; ++card) {
      const bool expected =
          (probe.legal_cards & core::card_bit(static_cast<core::CardId>(card))) != 0;
      const bool actual = has_torch_action50(
          packet.legal_actions,
          static_cast<std::uint16_t>(kTorch50ChoiceActionBegin + card));
      require(actual == expected, "post-chance legal mask mismatch");
    }
  }
}

void verify_features(const TorchPacket50& packet) {
  float features[kTorch50FeatureCount];
  encode_torch_packet50(packet, features);

  for (std::uint16_t i = 0; i < kTorch50SemanticFeatureCount; ++i) {
    require(std::isfinite(features[i]), "feature must be finite");
    require(features[i] >= 0.0f && features[i] <= 1.0f,
            "semantic feature must stay in [0, 1]");
  }
  for (std::uint16_t i = kTorch50SemanticFeatureCount;
       i < kTorch50FeatureCount; ++i) {
    require(features[i] == 0.0f, "padding features must be zero");
  }

  std::uint16_t decision_hot = 0;
  for (std::uint16_t i = 0; i < kTorch50DecisionCount; ++i) {
    decision_hot += static_cast<std::uint16_t>(
        features[kTorch50DecisionOffset + i] == 1.0f);
  }
  require(decision_hot == 1, "decision kind must be one-hot");
  require(features[kTorch50DecisionOffset + packet.decision_kind] == 1.0f,
          "decision kind one-hot index");
}

void test_feature_and_action_contract(std::uint64_t& packets,
                                      std::uint64_t& choice_packets) {
  constexpr std::uint32_t kSamples = 2048;
  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const std::uint64_t seed =
        0x510e527fade682d1ull +
        static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull;
    TorchEnv50 env = make_torch_env50(seed, static_cast<std::uint8_t>(i & 1u));

    for (std::uint16_t decision = 0; decision < 192 && env.done == 0; ++decision) {
      const TorchPacket50 packet = make_torch_packet50(env);
      verify_packet_contract(env);
      verify_features(packet);
      ++packets;
      if (env.pending.active != 0) ++choice_packets;

      const std::uint16_t action = canonical_torch_action50(env);
      require(action != kInvalidTorch50Action, "canonical torch action");
      const TorchStepResult50 step = torch_step50(env, action);
      require(step.status == ActionStatus50::kOk, "canonical torch step");
    }

    require(env.done != 0, "feature-contract rollout must finish");
    require(is_valid_terminal_game_state50(env.game),
            "feature-contract final state must be valid");
  }
}

void test_rollout_equivalence(std::uint64_t& primary_actions,
                              std::uint64_t& choice_actions,
                              std::uint64_t& resolve_choices,
                              std::uint64_t& pi_choices) {
  constexpr std::uint32_t kSamples = 8192;
  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const std::uint64_t seed =
        0x1f83d9abfb41bd6bull +
        static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull;
    const std::uint8_t first = static_cast<std::uint8_t>(i & 1u);

    const PostChanceRollout50 explicit_result =
        rollout_canonical_postchance50(seed, first);
    const TorchRollout50 torch_result = rollout_canonical_torch50(seed, first);

    require(equal_rollout_digest50(rollout_digest50(explicit_result.result),
                                   rollout_digest50(torch_result.result)),
            "torch rollout must match explicit post-chance rollout");
    require(torch_result.resolve_choices == explicit_result.resolve_choices,
            "resolve choice count must match explicit rollout");
    require(torch_result.pi_choices == explicit_result.pi_choices,
            "pi choice count must match explicit rollout");
    require(torch_result.result.end == RolloutEnd50::kTerminal ||
                torch_result.result.end == RolloutEnd50::kNagari,
            "torch rollout must finish");

    primary_actions += torch_result.result.actions;
    choice_actions += torch_result.choice_actions;
    resolve_choices += torch_result.resolve_choices;
    pi_choices += torch_result.pi_choices;
  }
}

}  // namespace

int main() {
  try {
    std::uint64_t packets = 0;
    std::uint64_t choice_packets = 0;
    std::uint64_t primary_actions = 0;
    std::uint64_t choice_actions = 0;
    std::uint64_t resolve_choices = 0;
    std::uint64_t pi_choices = 0;

    test_feature_and_action_contract(packets, choice_packets);
    test_rollout_equivalence(primary_actions, choice_actions,
                             resolve_choices, pi_choices);

    require(choice_packets != 0, "must exercise post-chance packets");
    require(resolve_choices != 0, "must exercise resolve choices");
    require(pi_choices != 0, "must exercise pi choices");

    std::cout << "cugo_torch50_test: PASS (8192 rollouts; packets=" << packets
              << " choice_packets=" << choice_packets
              << " primary_actions=" << primary_actions
              << " choice_actions=" << choice_actions
              << " resolve_choices=" << resolve_choices
              << " pi_choices=" << pi_choices
              << " features=" << kTorch50FeatureCount
              << " actions=" << kTorch50ActionCount << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cugo_torch50_test: FAIL: " << e.what() << '\n';
    return 1;
  }
}
