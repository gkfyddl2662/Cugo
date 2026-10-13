#pragma once

#include <cstdint>

#include "cugo/game/postchance50.h"

#if defined(__CUDACC__)
#define CUGO_HD __host__ __device__
#else
#define CUGO_HD
#endif

namespace cugo::game {

inline constexpr std::uint16_t kTorch50PrimaryActionCount = kPolicyActionCount50;
inline constexpr std::uint16_t kTorch50ChoiceActionBegin = kPolicyActionCount50;
inline constexpr std::uint16_t kTorch50ChoiceActionCount = core::kShinMatgoCardCount;
// Two Gukjin-prompt answers follow the card choices (Hangame Shin Matgo).
inline constexpr std::uint16_t kTorch50GukjinActionBegin =
    kTorch50PrimaryActionCount + kTorch50ChoiceActionCount;
inline constexpr std::uint16_t kTorch50ActionCount = kTorch50GukjinActionBegin + 2;
inline constexpr std::uint16_t kInvalidTorch50Action = 0xffffu;

inline constexpr std::uint16_t kTorch50CardPlaneCount = 6;
inline constexpr std::uint16_t kTorch50MonthPlaneCount = 8;
inline constexpr std::uint16_t kTorch50DecisionCount = 7;
inline constexpr std::uint16_t kTorch50ScalarCount = 26;
inline constexpr std::uint16_t kTorch50CardPlaneOffset = 0;
inline constexpr std::uint16_t kTorch50MonthPlaneOffset =
    kTorch50CardPlaneOffset +
    kTorch50CardPlaneCount * core::kShinMatgoCardCount;
inline constexpr std::uint16_t kTorch50SelectedPiOffset =
    kTorch50MonthPlaneOffset + kTorch50MonthPlaneCount * core::kMonthCount;
inline constexpr std::uint16_t kTorch50DecisionOffset =
    kTorch50SelectedPiOffset + core::kShinMatgoCardCount;
inline constexpr std::uint16_t kTorch50ScalarOffset =
    kTorch50DecisionOffset + kTorch50DecisionCount;
inline constexpr std::uint16_t kTorch50SemanticFeatureCount =
    kTorch50ScalarOffset + kTorch50ScalarCount;
// Hangame extensions are appended after the original 496 (479 used + padding) so
// older networks can be widened with zero weights.
inline constexpr std::uint16_t kTorch50MissionPlaneOffset = 496;
inline constexpr std::uint16_t kTorch50ExtScalarOffset =
    kTorch50MissionPlaneOffset + core::kStandardCardCount;
inline constexpr std::uint16_t kTorch50ExtScalarCount = 16;
// Contract v3: the neutral-ppuk month plane takes 12 of the legacy padding slots, and
// three public-history card planes (own played, opponent played, opponent cards
// known from a shake) follow the Hangame block. Zero padding keeps the width a
// multiple of 16.
inline constexpr std::uint16_t kTorch50NeutralPpukOffset = kTorch50ScalarOffset + kTorch50ScalarCount;
inline constexpr std::uint16_t kTorch50HistoryOffset =
    kTorch50ExtScalarOffset + kTorch50ExtScalarCount;
inline constexpr std::uint16_t kTorch50HistoryPlaneCount = 3;
inline constexpr std::uint16_t kTorch50FeatureCount = 720;

static_assert(kTorch50PrimaryActionCount == 127);
static_assert(kTorch50ActionCount == 179);
static_assert(kTorch50GukjinActionBegin == kPolicyGukjinPi50);
static_assert(kTorch50GukjinActionBegin + 1u == kPolicyGukjinAnimal50);
static_assert(kTorch50HistoryOffset == 560);
static_assert(kTorch50SemanticFeatureCount == 479);
static_assert((kTorch50FeatureCount % 16u) == 0u);
static_assert(kTorch50NeutralPpukOffset + core::kMonthCount <= kTorch50MissionPlaneOffset);
static_assert(kTorch50HistoryOffset + kTorch50HistoryPlaneCount * core::kShinMatgoCardCount <=
              kTorch50FeatureCount);

struct TorchActionMask50 {
  std::uint64_t word0;
  std::uint64_t word1;
  std::uint64_t word2;
};

struct TorchPacket50 {
  PolicyObservation50 observation;
  TorchActionMask50 legal_actions;
  core::CardMask selected_pi;
  std::uint16_t primary_action;
  core::CardId revealed_card;
  std::uint8_t requested_pi;
  std::uint8_t decision_player;
  std::uint8_t decision_kind;
  ActionStatus50 status;
};

struct TorchEnv50 {
  TerminalGameState50 game;
  PostChancePending50 pending;
  std::uint16_t primary_actions;
  std::uint16_t choice_actions;
  std::uint16_t resolve_choices;
  std::uint16_t pi_choices;
  std::uint16_t bonus_actions;
  std::uint16_t special_actions;
  std::uint16_t go_actions;
  ActionStatus50 status;
  std::uint8_t done;
  std::uint8_t nagari;
};

struct TorchStepResult50 {
  ActionStatus50 status;
  std::int64_t reward0;
  std::uint8_t done;
  std::uint8_t nagari;
  std::uint8_t primary_committed;
};

struct TorchRollout50 {
  RolloutResult50 result;
  std::uint16_t choice_actions;
  std::uint16_t resolve_choices;
  std::uint16_t pi_choices;
};

CUGO_HD inline TorchActionMask50 empty_torch_action_mask50() noexcept {
  return TorchActionMask50{0, 0, 0};
}

CUGO_HD inline void set_torch_action50(TorchActionMask50& mask,
                                        std::uint16_t action) noexcept {
  if (action >= kTorch50ActionCount) return;
  const std::uint64_t bit = std::uint64_t{1} << (action & 63u);
  if (action < 64u) mask.word0 |= bit;
  else if (action < 128u) mask.word1 |= bit;
  else mask.word2 |= bit;
}

CUGO_HD inline bool has_torch_action50(TorchActionMask50 mask,
                                       std::uint16_t action) noexcept {
  if (action >= kTorch50ActionCount) return false;
  const std::uint64_t bit = std::uint64_t{1} << (action & 63u);
  if (action < 64u) return (mask.word0 & bit) != 0;
  if (action < 128u) return (mask.word1 & bit) != 0;
  return (mask.word2 & bit) != 0;
}

CUGO_HD inline std::uint16_t torch_action_count50(
    TorchActionMask50 mask) noexcept {
  return static_cast<std::uint16_t>(core::card_count(mask.word0) +
                                    core::card_count(mask.word1) +
                                    core::card_count(mask.word2));
}

CUGO_HD inline TorchActionMask50 primary_torch_action_mask50(
    PolicyActionMask50 primary) noexcept {
  return TorchActionMask50{primary.lo, primary.hi, 0};
}

CUGO_HD inline TorchActionMask50 choice_torch_action_mask50(
    core::CardMask cards) noexcept {
  TorchActionMask50 mask = empty_torch_action_mask50();
  cards &= core::kShinMatgoDeckMask;
  while (cards != 0) {
    const core::CardId card = core::pop_first_card(cards);
    set_torch_action50(mask,
                       static_cast<std::uint16_t>(kTorch50ChoiceActionBegin + card));
  }
  return mask;
}

CUGO_HD inline std::uint8_t torch_decision_kind50(
    const TorchEnv50& env) noexcept {
  if (env.pending.active != 0) {
    TerminalGameState50 staged{};
    const PostChanceProbe50 probe =
        probe_postchance50(env.game, env.pending, staged);
    if (probe.status != ActionStatus50::kOk) return 0;
    return static_cast<std::uint8_t>(probe.decision);
  }
  return static_cast<std::uint8_t>(policy_decision_kind50(env.game));
}

CUGO_HD inline bool torch_nagari_ready50(const TorchEnv50& env) noexcept {
  return env.pending.active == 0 && !terminal50_is_finished(env.game) &&
         !terminal50_has_chongtong_choice(env.game) &&
         !game50_has_pending_choice(env.game.special.game) &&
         rollout_hands_exhausted50(env.game);
}

CUGO_HD inline void refresh_torch_done50(TorchEnv50& env) noexcept {
  if (terminal50_is_finished(env.game)) {
    env.done = 1;
    env.nagari = 0;
    return;
  }
  if (torch_nagari_ready50(env)) {
    env.done = 1;
    env.nagari = 1;
    return;
  }
  env.done = 0;
  env.nagari = 0;
}

CUGO_HD inline TorchEnv50 make_torch_env50(
    InitialDeal50 deal,
    std::uint8_t first_player = 0,
    std::uint8_t round_multiplier = 1) noexcept {
  TorchEnv50 env{make_terminal_game_state50(deal, first_player, round_multiplier),
                 PostChancePending50{},
                 0, 0, 0, 0, 0, 0, 0,
                 ActionStatus50::kOk,
                 0, 0};
  refresh_torch_done50(env);
  return env;
}

CUGO_HD inline TorchEnv50 make_torch_env50(
    std::uint64_t seed,
    std::uint8_t first_player = 0,
    std::uint8_t round_multiplier = 1) noexcept {
  return make_torch_env50(deal_shin_matgo_50(seed), first_player, round_multiplier);
}

CUGO_HD inline TorchActionMask50 gukjin_torch_action_mask50() noexcept {
  TorchActionMask50 mask = empty_torch_action_mask50();
  set_torch_action50(mask, kTorch50GukjinActionBegin);
  set_torch_action50(mask, static_cast<std::uint16_t>(kTorch50GukjinActionBegin + 1u));
  return mask;
}

CUGO_HD inline TorchPacket50 make_torch_packet50(
    const TorchEnv50& env) noexcept {
  if (env.pending.active != 0) {
    TerminalGameState50 staged{};
    const PostChanceProbe50 probe =
        probe_postchance50(env.game, env.pending, staged);
    if (probe.status != ActionStatus50::kOk) {
      return TorchPacket50{PolicyObservation50{},
                           empty_torch_action_mask50(),
                           env.pending.selected_pi,
                           env.pending.primary_action,
                           probe.revealed_card,
                           probe.requested_pi,
                           env.pending.actor,
                           static_cast<std::uint8_t>(PostChanceDecision50::kNone),
                           probe.status};
    }
    const PostChancePacket50 packet =
        make_postchance_packet50(env.game, staged, env.pending, probe);
    return TorchPacket50{packet.observation,
                         choice_torch_action_mask50(packet.legal_cards),
                         packet.selected_pi,
                         packet.primary_action,
                         packet.revealed_card,
                         packet.requested_pi,
                         packet.decision_player,
                         static_cast<std::uint8_t>(packet.decision),
                         ActionStatus50::kOk};
  }

  const PolicyPacket50 packet = make_policy_packet50(env.game);
  std::uint8_t player = policy_decision_player50(env.game);
  if (player > 1u) {
    player = static_cast<std::uint8_t>(env.game.special.game.turn.actor & 1u);
  }
  const bool gukjin = packet.observation.decision_kind ==
                     static_cast<std::uint8_t>(PolicyDecision50::kGukjin);
  return TorchPacket50{packet.observation,
                       env.done != 0
                           ? empty_torch_action_mask50()
                           : gukjin ? gukjin_torch_action_mask50()
                                    : primary_torch_action_mask50(packet.legal_actions),
                       0,
                       kInvalidPolicyAction50,
                       core::kInvalidCard,
                       0,
                       player,
                       env.done != 0
                           ? static_cast<std::uint8_t>(0)
                           : packet.observation.decision_kind,
                       ActionStatus50::kOk};
}

CUGO_HD inline float torch_unit50(std::uint32_t value,
                                  std::uint32_t scale) noexcept {
  if (scale == 0) return 0.0f;
  const float result = static_cast<float>(value) / static_cast<float>(scale);
  return result > 1.0f ? 1.0f : result;
}

CUGO_HD inline void encode_torch_card_plane50(float* features,
                                               std::uint16_t offset,
                                               core::CardMask cards) noexcept {
  for (std::uint16_t card = 0; card < core::kShinMatgoCardCount; ++card) {
    features[offset + card] =
        (cards & (std::uint64_t{1} << card)) != 0 ? 1.0f : 0.0f;
  }
}

CUGO_HD inline void encode_torch_month_plane50(float* features,
                                                std::uint16_t offset,
                                                std::uint16_t months) noexcept {
  for (std::uint16_t month = 0; month < core::kMonthCount; ++month) {
    features[offset + month] =
        (months & (std::uint16_t{1} << month)) != 0 ? 1.0f : 0.0f;
  }
}

CUGO_HD inline std::uint16_t torch_choice_action_count50(
    TorchActionMask50 mask) noexcept {
  std::uint16_t count = 0;
  for (std::uint16_t card = 0; card < core::kShinMatgoCardCount; ++card) {
    count += static_cast<std::uint16_t>(
        has_torch_action50(mask,
                           static_cast<std::uint16_t>(kTorch50ChoiceActionBegin + card)));
  }
  return count;
}

CUGO_HD inline void encode_torch_packet50(const TorchPacket50& packet,
                                           float* features) noexcept {
  for (std::uint16_t i = 0; i < kTorch50FeatureCount; ++i) features[i] = 0.0f;

  const PolicyObservation50& o = packet.observation;
  const core::CardMask card_planes[kTorch50CardPlaneCount] = {
      o.own_hand,
      o.floor,
      o.own_captured,
      o.opponent_captured,
      o.pending_public,
      o.unseen};
  for (std::uint16_t plane = 0; plane < kTorch50CardPlaneCount; ++plane) {
    encode_torch_card_plane50(
        features,
        static_cast<std::uint16_t>(kTorch50CardPlaneOffset +
                                   plane * core::kShinMatgoCardCount),
        card_planes[plane]);
  }

  const std::uint16_t month_planes[kTorch50MonthPlaneCount] = {
      o.own_shaken_months,
      o.opponent_shaken_months,
      o.ppuk_months,
      o.own_ppuk_months,
      o.opponent_ppuk_months,
      o.bonus2_ppuk_months,
      o.bonus3_ppuk_months,
      o.pending_chongtong_months};
  for (std::uint16_t plane = 0; plane < kTorch50MonthPlaneCount; ++plane) {
    encode_torch_month_plane50(
        features,
        static_cast<std::uint16_t>(kTorch50MonthPlaneOffset +
                                   plane * core::kMonthCount),
        month_planes[plane]);
  }

  encode_torch_card_plane50(features, kTorch50SelectedPiOffset,
                            packet.selected_pi);
  if (packet.decision_kind < kTorch50DecisionCount) {
    features[kTorch50DecisionOffset + packet.decision_kind] = 1.0f;
  }

  float* s = features + kTorch50ScalarOffset;
  s[0] = torch_unit50(o.turn_index, 32);
  s[1] = torch_unit50(o.stock_count, 30);
  s[2] = torch_unit50(o.own_go_count, 8);
  s[3] = torch_unit50(o.opponent_go_count, 8);
  s[4] = torch_unit50(o.own_last_go_base_score, 64);
  s[5] = torch_unit50(o.opponent_last_go_base_score, 64);
  s[6] = torch_unit50(o.own_base_score, 64);
  s[7] = torch_unit50(o.opponent_base_score, 64);
  s[8] = torch_unit50(o.own_bombs, 4);
  s[9] = torch_unit50(o.opponent_bombs, 4);
  s[10] = torch_unit50(o.own_credits, 4);
  s[11] = torch_unit50(o.opponent_credits, 4);
  s[12] = torch_unit50(o.own_ppuk_count, 3);
  s[13] = torch_unit50(o.opponent_ppuk_count, 3);
  s[14] = torch_unit50(o.own_ppuk_streak <= 3u ? o.own_ppuk_streak : 0u, 3);
  s[15] = torch_unit50(o.opponent_ppuk_streak <= 3u ? o.opponent_ppuk_streak : 0u, 3);
  s[16] = o.own_gukjin_double_pi != 0 ? 1.0f : 0.0f;
  s[17] = o.opponent_gukjin_double_pi != 0 ? 1.0f : 0.0f;
  s[18] = torch_unit50(packet.requested_pi, 8);
  s[19] = packet.primary_action < kPolicyActionCount50
      ? static_cast<float>(packet.primary_action + 1u) /
            static_cast<float>(kPolicyActionCount50)
      : 0.0f;
  s[20] = core::is_physical_card(packet.revealed_card)
      ? static_cast<float>(packet.revealed_card + 1u) /
            static_cast<float>(core::kShinMatgoCardCount)
      : 0.0f;
  s[21] = torch_unit50(
      static_cast<std::uint32_t>(core::card_count(packet.selected_pi)), 50);
  s[22] = torch_unit50(torch_choice_action_count50(packet.legal_actions), 50);
  s[23] = torch_unit50(
      static_cast<std::uint32_t>(core::card_count(o.own_hand)), 50);
  s[24] = torch_unit50(
      static_cast<std::uint32_t>(core::card_count(o.floor)), 50);
  s[25] = torch_unit50(
      static_cast<std::uint32_t>(core::card_count(o.unseen)), 50);

  for (std::uint16_t card = 0; card < core::kStandardCardCount; ++card) {
    features[kTorch50MissionPlaneOffset + card] =
        (o.mission_cards & (std::uint64_t{1} << card)) != 0 ? 1.0f : 0.0f;
  }
  float* x = features + kTorch50ExtScalarOffset;
  const std::uint8_t mission = o.mission_type;
  x[0] = mission == kMissionEasy50 ? 1.0f : 0.0f;
  x[1] = mission == kMissionNormal50 ? 1.0f : 0.0f;
  x[2] = mission == kMissionHard50 ? 1.0f : 0.0f;
  x[3] = mission == kMissionNoCondition50 ? 1.0f : 0.0f;
  x[4] = mission == kMissionTriple50 ? 1.0f : 0.0f;
  x[5] = torch_unit50(o.mission_multiplier, 4);
  x[6] = torch_unit50(o.own_mission_multi, 8);
  x[7] = torch_unit50(o.opponent_mission_multi, 8);
  x[8] = packet.decision_kind == static_cast<std::uint8_t>(PolicyDecision50::kGukjin)
      ? 1.0f : 0.0f;
  x[9] = o.own_gukjin_locked != 0 ? 1.0f : 0.0f;
  x[10] = o.opponent_gukjin_locked != 0 ? 1.0f : 0.0f;
  x[11] = o.last_turn != 0 ? 1.0f : 0.0f;
  const float side = static_cast<float>(o.own_side_points) / 64.0f;
  x[12] = side > 1.0f ? 1.0f : side < -1.0f ? -1.0f : side;
  x[13] = torch_unit50(o.round_multiplier, 8);
  x[14] = o.own_ppuk_streak == kPpukFirst50 || o.own_ppuk_streak == kPpukSecond50
      ? 1.0f : 0.0f;
  x[15] = o.opponent_ppuk_streak == kPpukFirst50 || o.opponent_ppuk_streak == kPpukSecond50
      ? 1.0f : 0.0f;

  encode_torch_month_plane50(features, kTorch50NeutralPpukOffset, o.neutral_ppuk_months);
  const core::CardMask history[kTorch50HistoryPlaneCount] = {
      o.own_played, o.opponent_played, o.opponent_known_hand};
  for (std::uint16_t plane = 0; plane < kTorch50HistoryPlaneCount; ++plane) {
    encode_torch_card_plane50(
        features,
        static_cast<std::uint16_t>(kTorch50HistoryOffset + plane * core::kShinMatgoCardCount),
        history[plane]);
  }
}

CUGO_HD inline std::int64_t torch_reward0_50(const TorchEnv50& env) noexcept {
  if (env.done == 0) return 0;
  if (env.nagari != 0) return nagari_reward50(env.game, 0);
  if (!terminal50_is_finished(env.game)) return 0;
  const Settlement50 settlement = settle_terminal50(env.game);
  return settlement_reward50(settlement, 0);
}

CUGO_HD inline void record_torch_primary50(TorchEnv50& env,
                                            ActionKind50 kind) noexcept {
  ++env.primary_actions;
  if (kind == ActionKind50::kPlayBonus) ++env.bonus_actions;
  if (kind == ActionKind50::kShake || kind == ActionKind50::kBomb ||
      kind == ActionKind50::kGrenade || kind == ActionKind50::kBombCredit) {
    ++env.special_actions;
  }
  if (kind == ActionKind50::kGo) ++env.go_actions;
}

CUGO_HD inline TorchStepResult50 torch_step50(TorchEnv50& env,
                                               std::uint16_t action) noexcept {
  if (env.done != 0) {
    env.status = ActionStatus50::kGameFinished;
    return TorchStepResult50{env.status, torch_reward0_50(env), env.done,
                             env.nagari, 0};
  }

  const TorchPacket50 packet = make_torch_packet50(env);
  if (packet.status != ActionStatus50::kOk) {
    env.status = packet.status;
    return TorchStepResult50{env.status, 0, env.done, env.nagari, 0};
  }
  if (!has_torch_action50(packet.legal_actions, action)) {
    env.status = ActionStatus50::kNotLegal;
    return TorchStepResult50{env.status, 0, env.done, env.nagari, 0};
  }

  const bool choosing = env.pending.active != 0;
  PostChanceStep50 step{};
  if (choosing) {
    const core::CardId card = static_cast<core::CardId>(
        action - kTorch50ChoiceActionBegin);
    if (packet.decision_kind ==
        static_cast<std::uint8_t>(PostChanceDecision50::kPiCard)) {
      ++env.pi_choices;
    } else {
      ++env.resolve_choices;
    }
    ++env.choice_actions;
    step = choose_postchance_card50(env.game, env.pending, card);
  } else {
    step = start_postchance_action50(env.game, action, env.pending);
  }

  if (step.status != ActionStatus50::kOk) {
    env.status = step.status;
    return TorchStepResult50{env.status, 0, env.done, env.nagari, 0};
  }

  std::uint8_t committed = 0;
  if (step.action_committed != 0) {
    record_torch_primary50(env, env.pending.action.kind);
    committed = 1;
  }
  env.status = ActionStatus50::kOk;
  refresh_torch_done50(env);
  return TorchStepResult50{env.status, torch_reward0_50(env), env.done,
                           env.nagari, committed};
}

CUGO_HD inline std::uint16_t canonical_torch_action50(
    const TorchEnv50& env) noexcept {
  if (env.done != 0) return kInvalidTorch50Action;
  if (env.pending.active != 0) {
    const TorchPacket50 packet = make_torch_packet50(env);
    if (packet.status != ActionStatus50::kOk) return kInvalidTorch50Action;
    for (std::uint16_t card = 0; card < core::kShinMatgoCardCount; ++card) {
      const std::uint16_t action =
          static_cast<std::uint16_t>(kTorch50ChoiceActionBegin + card);
      if (has_torch_action50(packet.legal_actions, action)) return action;
    }
    return kInvalidTorch50Action;
  }

  const CanonicalAction50 canonical = canonical_action50(env.game);
  if (canonical.status != ActionStatus50::kOk) return kInvalidTorch50Action;
  return primary_action_index50(canonical.action);
}

CUGO_HD inline TorchRollout50 finish_torch_rollout50(
    const TorchEnv50& env,
    RolloutEnd50 fallback_end = RolloutEnd50::kMaxActions) noexcept {
  RolloutResult50 result{};
  if (terminal50_is_finished(env.game)) {
    result = finish_terminal_rollout50(env.game, env.primary_actions,
                                       env.bonus_actions, env.special_actions,
                                       env.go_actions);
  } else if (env.nagari != 0) {
    result = RolloutResult50{RolloutEnd50::kNagari,
                             ActionStatus50::kOk,
                             TerminalReason50::kNone,
                             kNoTerminal50Player,
                             2,
                             env.primary_actions,
                             env.game.special.game.turn.turn_index,
                             env.bonus_actions,
                             env.special_actions,
                             env.go_actions,
                             0,
                             nagari_reward50(env.game, 0)};
  } else {
    result = RolloutResult50{fallback_end,
                             env.status,
                             TerminalReason50::kNone,
                             kNoTerminal50Player,
                             1,
                             env.primary_actions,
                             env.game.special.game.turn.turn_index,
                             env.bonus_actions,
                             env.special_actions,
                             env.go_actions,
                             0,
                             0};
  }
  return TorchRollout50{result, env.choice_actions, env.resolve_choices,
                        env.pi_choices};
}

CUGO_HD inline TorchRollout50 rollout_canonical_torch50(
    std::uint64_t seed,
    std::uint8_t first_player = 0,
    std::uint16_t max_actions = 128) noexcept {
  TorchEnv50 env = make_torch_env50(seed, first_player);
  while (env.done == 0 && env.primary_actions < max_actions) {
    const std::uint16_t action = canonical_torch_action50(env);
    if (action == kInvalidTorch50Action) {
      env.status = ActionStatus50::kInvalidAction;
      return finish_torch_rollout50(env, RolloutEnd50::kStalled);
    }
    const TorchStepResult50 step = torch_step50(env, action);
    if (step.status != ActionStatus50::kOk) {
      return finish_torch_rollout50(env, RolloutEnd50::kActionError);
    }
  }
  return finish_torch_rollout50(env,
                                env.done != 0 ? RolloutEnd50::kTerminal
                                              : RolloutEnd50::kMaxActions);
}

// ---------------------------------------------------------------------------
// Event log: one record per committed action, the table-wide difference between
// the states before and after it. Records are observer-independent; whoever
// encodes them for a player must hide `drawn` unless that player is the actor.
// Replaying the records from the start record reproduces the floor and both
// captured piles exactly.

inline constexpr std::uint16_t kTorch50EventWords = 9;
inline constexpr std::uint16_t kTorch50StartWords = 8;

struct TorchEvent50 {
  core::CardMask played;   // left the actor's hand (public)
  core::CardMask drawn;    // entered the actor's hand from the stock (actor only)
  core::CardMask flipped;  // left the stock for the table (public)
  core::CardMask gained0;  // entered player 0's captured pile
  core::CardMask gained1;
  core::CardMask lost0;    // left player 0's captured pile (stolen)
  core::CardMask lost1;
  core::CardMask shown;    // shown by a shake declaration
  std::uint8_t actor;
  std::uint8_t kind;       // ActionKind50
};

CUGO_HD inline TorchEvent50 make_torch_event50(const TorchEnv50& before,
                                               const TorchEnv50& after) noexcept {
  const TurnState50& b = before.game.special.game.turn;
  const TurnState50& a = after.game.special.game.turn;
  const std::uint8_t actor = static_cast<std::uint8_t>(after.pending.actor & 1u);
  const core::CardMask hand_before = actor == 0 ? b.hand0 : b.hand1;
  const core::CardMask hand_after = actor == 0 ? a.hand0 : a.hand1;
  const core::CardMask revealed_before = actor == 0 ? before.game.revealed0 : before.game.revealed1;
  const core::CardMask revealed_after = actor == 0 ? after.game.revealed0 : after.game.revealed1;
  TorchEvent50 e{};
  e.played = hand_before & ~hand_after;
  e.drawn = hand_after & ~hand_before;
  e.flipped = b.stock & ~a.stock & ~e.drawn;
  e.gained0 = a.captured0 & ~b.captured0;
  e.gained1 = a.captured1 & ~b.captured1;
  e.lost0 = b.captured0 & ~a.captured0;
  e.lost1 = b.captured1 & ~a.captured1;
  e.shown = revealed_after & ~revealed_before;
  e.actor = actor;
  e.kind = static_cast<std::uint8_t>(after.pending.action.kind);
  return e;
}

// Words 0..7 are the card masks in field order; word 8 packs actor | kind << 8.
CUGO_HD inline void pack_torch_event50(const TorchEvent50& e, std::int64_t* out) noexcept {
  out[0] = static_cast<std::int64_t>(e.played);
  out[1] = static_cast<std::int64_t>(e.drawn);
  out[2] = static_cast<std::int64_t>(e.flipped);
  out[3] = static_cast<std::int64_t>(e.gained0);
  out[4] = static_cast<std::int64_t>(e.gained1);
  out[5] = static_cast<std::int64_t>(e.lost0);
  out[6] = static_cast<std::int64_t>(e.lost1);
  out[7] = static_cast<std::int64_t>(e.shown);
  out[8] = static_cast<std::int64_t>(e.actor) | (static_cast<std::int64_t>(e.kind) << 8);
}

// The game's opening as each player learns it: both hands (encoders show only the
// observer's), the refilled floor, floor bonuses already captured, and the mission.
// Word 6 is the neutral-ppuk months; word 7 packs mission type | multiplier << 8 | first player << 16 |
// mission_multi0 << 24 | mission_multi1 << 32.
CUGO_HD inline void pack_torch_start50(const TorchEnv50& env, std::int64_t* out) noexcept {
  const TurnState50& t = env.game.special.game.turn;
  out[0] = static_cast<std::int64_t>(t.hand0);
  out[1] = static_cast<std::int64_t>(t.hand1);
  out[2] = static_cast<std::int64_t>(t.floor);
  out[3] = static_cast<std::int64_t>(t.captured0);
  out[4] = static_cast<std::int64_t>(t.captured1);
  out[5] = static_cast<std::int64_t>(t.mission_cards);
  out[6] = static_cast<std::int64_t>(t.neutral_ppuk_months);
  out[7] = static_cast<std::int64_t>(env.game.mission_type) |
           (static_cast<std::int64_t>(env.game.mission_multiplier) << 8) |
           (static_cast<std::int64_t>(t.actor & 1u) << 16) |
           (static_cast<std::int64_t>(env.game.mission_multi0) << 24) |
           (static_cast<std::int64_t>(env.game.mission_multi1) << 32);
}

// Steps the environment and reports the committed action's event, if any.
CUGO_HD inline TorchStepResult50 torch_step_event50(TorchEnv50& env, std::uint16_t action,
                                                    TorchEvent50& event) noexcept {
  const TorchEnv50 before = env;
  const TorchStepResult50 result = torch_step50(env, action);
  event = result.primary_committed != 0 ? make_torch_event50(before, env) : TorchEvent50{};
  return result;
}

}  // namespace cugo::game

#undef CUGO_HD
