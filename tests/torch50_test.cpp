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
  for (std::uint16_t i = kTorch50TableCountOffset; i < kTorch50MissionPlaneOffset; ++i) {
    require(features[i] >= 0.0f && features[i] <= 1.0f, "table counts stay in [0, 1]");
  }
  constexpr std::uint16_t kHistoryEnd =
      kTorch50HistoryOffset + kTorch50HistoryPlaneCount * core::kShinMatgoCardCount;
  for (std::uint16_t i = kTorch50RibbonCountOffset + 2; i < kTorch50FeatureCount; ++i) {
    require(features[i] == 0.0f, "trailing padding must be zero");
  }
  for (std::uint16_t card = 0; card < core::kShinMatgoCardCount; ++card) {
    require(features[kTorch50ChoiceCardOffset + card] ==
                (has_torch_action50(packet.legal_actions,
                                    static_cast<std::uint16_t>(kTorch50ChoiceActionBegin + card))
                     ? 1.0f : 0.0f),
            "choice-card plane mirrors the legal floor choices");
  }
  for (std::uint16_t i = kTorch50HistoryOffset; i < kHistoryEnd; ++i) {
    require(features[i] == 0.0f || features[i] == 1.0f, "history planes are binary");
  }

  // Public-history planes never overlap what they claim to exclude.
  const PolicyObservation50& o = packet.observation;
  require((o.own_played & o.own_hand) == 0, "own played cards are out of hand");
  require((o.opponent_known_hand & (o.own_hand | o.floor | o.own_captured |
                                     o.opponent_captured | o.unseen)) == 0,
          "known opponent cards are neither public nor unseen");
  require((o.own_played & o.opponent_played) == 0, "a card is played by one side only");

  // Every card is in exactly one place from the observer's seat.
  const core::CardMask places[] = {o.own_hand, o.floor, o.own_captured, o.opponent_captured,
                                   o.pending_public, o.unseen, o.opponent_known_hand};
  core::CardMask seen = 0;
  for (const core::CardMask place : places) {
    require((seen & place) == 0, "a card is in one place only");
    seen |= place;
  }
  require(seen == core::kShinMatgoDeckMask, "every card has a place");

  // Kinds 1..5 are one-hot; slots 0 and 6 carry the first-player bit and the
  // opponent's hand count (contract v4).
  std::uint16_t decision_hot = 0;
  for (std::uint16_t i = 1; i <= 5; ++i) {
    decision_hot += static_cast<std::uint16_t>(
        features[kTorch50DecisionOffset + i] == 1.0f);
  }
  require(features[kTorch50OwnFirstIndex] == (o.own_first != 0 ? 1.0f : 0.0f), "first-player bit");
  if (packet.decision_kind >= 1u && packet.decision_kind <= 5u) {
    require(decision_hot == 1, "decision kind must be one-hot");
    require(features[kTorch50DecisionOffset + packet.decision_kind] == 1.0f,
            "decision kind one-hot index");
    require(features[kTorch50ExtScalarOffset + 8] == 0.0f,
            "gukjin flag only on the gukjin prompt");
  } else {
    // The gukjin prompt sits outside the legacy one-hot; an appended flag marks it.
    require(packet.decision_kind ==
                static_cast<std::uint8_t>(PolicyDecision50::kGukjin),
            "unknown decision kind");
    require(decision_hot == 0, "gukjin prompt leaves the legacy one-hot empty");
    require(features[kTorch50ExtScalarOffset + 8] == 1.0f,
            "gukjin prompt flag");
  }
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

// Random play: the v4 observation facts against ground truth tracked outside the
// engine (first player, cards each hand has held, the stop settlement).
void test_observation_facts(std::uint64_t& floor_choices, std::uint64_t& go_stops) {
  std::uint64_t rng = 0x6f62736572766535ULL;
  auto next = [&rng]() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(rng >> 33);
  };
  for (std::uint32_t g = 0; g < 8192; ++g) {
    const std::uint8_t first = static_cast<std::uint8_t>(g & 1u);
    TorchEnv50 env = make_torch_env50(0x9b05688c2b3e6c1fULL + 0x9e3779b97f4a7c15ULL * g, first);
    if (env.game.mission_type == kMissionTriple50) {
      require(env.game.mission_multi0 == 1 && env.game.mission_multi1 == 1,
              "triple mission starts at 2^0");
    }
    const TurnState50& t0 = env.game.special.game.turn;
    core::CardMask held[2] = {t0.hand0, t0.hand1};
    for (std::uint16_t step = 0; step < 192 && env.done == 0; ++step) {
      const TorchPacket50 packet = make_torch_packet50(env);
      require(packet.status == ActionStatus50::kOk, "facts: packet");
      const PolicyObservation50& o = packet.observation;
      const std::uint8_t me = packet.decision_player;
      const TurnState50& t = env.game.special.game.turn;
      const core::CardMask opp_hand = me == 0 ? t.hand1 : t.hand0;
      require(o.own_first == static_cast<std::uint8_t>(me == first), "facts: first player");
      require(o.opponent_hand_count == core::card_count(opp_hand), "facts: opponent hand count");
      require(o.own_played == (held[me] & ~o.own_hand), "facts: own played cards");
      require(o.opponent_played == (held[me ^ 1u] & ~opp_hand), "facts: opponent played cards");
      if (env.pending.active != 0) ++floor_choices;
      if (packet.decision_kind == static_cast<std::uint8_t>(PolicyDecision50::kGoStop)) {
        ++go_stops;
        TerminalGameState50 stopped = env.game;
        require(apply_terminal_go_stop_decision50(stopped, GoStopAction50::kStop) ==
                    GoStopStatus50::kOk, "facts: stop applies");
        const Settlement50 settled = settle_terminal50(stopped);
        require(settled.winner == me && settled.final_points == o.own_stop_points,
                "facts: stop points are the stop settlement");
      } else {
        require(o.own_stop_points == 0, "facts: stop points only at Go/Stop");
      }

      std::uint16_t legal[kTorch50ActionCount];
      std::uint16_t n = 0;
      for (std::uint16_t a = 0; a < kTorch50ActionCount; ++a)
        if (has_torch_action50(packet.legal_actions, a)) legal[n++] = a;
      require(n != 0, "facts: legal action");
      const TorchStepResult50 r = torch_step50(env, legal[next() % n]);
      require(r.status == ActionStatus50::kOk, "facts: step");
      held[0] |= env.game.special.game.turn.hand0;
      held[1] |= env.game.special.game.turn.hand1;
    }
  }
}

}  // namespace

// Replaying the event log from the start record must reproduce the floor, both
// captured piles and the actor's hand after every committed action.
void test_event_log_replays_table(std::uint64_t& events) {
  std::uint64_t rng = 0x6576656e74733530ULL;
  auto next = [&rng]() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(rng >> 33);
  };
  for (std::uint32_t g = 0; g < 4096; ++g) {
    TorchEnv50 env = make_torch_env50(0x243f6a8885a308d3ULL + 0x9e3779b97f4a7c15ULL * g,
                                      static_cast<std::uint8_t>(g & 1u));
    std::int64_t start[kTorch50StartWords];
    pack_torch_start50(env, start);
    core::CardMask hand[2] = {static_cast<core::CardMask>(start[0]),
                              static_cast<core::CardMask>(start[1])};
    core::CardMask floor = static_cast<core::CardMask>(start[2]);
    core::CardMask captured[2] = {static_cast<core::CardMask>(start[3]),
                                  static_cast<core::CardMask>(start[4])};
    std::uint32_t count = 0;
    for (std::uint16_t step = 0; step < 192 && env.done == 0; ++step) {
      const TorchPacket50 packet = make_torch_packet50(env);
      std::uint16_t legal[kTorch50ActionCount];
      std::uint16_t n = 0;
      for (std::uint16_t a = 0; a < kTorch50ActionCount; ++a)
        if (has_torch_action50(packet.legal_actions, a)) legal[n++] = a;
      require(n != 0, "event replay: legal action");
      TorchEvent50 e{};
      const TorchStepResult50 r = torch_step_event50(env, legal[next() % n], e);
      require(r.status == ActionStatus50::kOk, "event replay: step");
      if (r.primary_committed == 0) continue;
      ++count;
      std::int64_t words[kTorch50EventWords];
      pack_torch_event50(e, words);
      require((words[8] & 0xff) == e.actor && ((words[8] >> 8) & 0xff) == e.kind,
              "event header packs actor and kind");
      hand[e.actor] = (hand[e.actor] & ~e.played) | e.drawn;
      floor = (floor | e.played | e.flipped) & ~e.gained0 & ~e.gained1;
      captured[0] = (captured[0] & ~e.lost0) | e.gained0;
      captured[1] = (captured[1] & ~e.lost1) | e.gained1;
      const TurnState50& t = env.game.special.game.turn;
      require(hand[0] == t.hand0 && hand[1] == t.hand1, "event replay: hands");
      require(floor == t.floor, "event replay: floor");
      require(captured[0] == t.captured0 && captured[1] == t.captured1,
              "event replay: captured piles");
      require((e.shown & ~(e.played | hand[e.actor])) == 0, "shown cards were in hand");
    }
    require(count <= 48, "event log fits 48 records");
    events += count;
  }
}

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
    std::uint64_t events = 0;
    test_event_log_replays_table(events);
    require(events > 4096 * 10, "event replay covers whole games");
    std::uint64_t floor_choices = 0;
    std::uint64_t go_stops = 0;
    test_observation_facts(floor_choices, go_stops);
    require(floor_choices > 1000 && go_stops > 1000, "facts cover floor choices and Go/Stop");

    require(choice_packets != 0, "must exercise post-chance packets");
    require(resolve_choices != 0, "must exercise resolve choices");
    // Hangame picks the stolen pi itself, so no pi-card choice is ever offered.
    require(pi_choices == 0, "pi steals must be automatic");

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
