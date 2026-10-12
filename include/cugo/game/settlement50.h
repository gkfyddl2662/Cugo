#pragma once

#include <cstdint>

#include "cugo/game/terminal50.h"

#if defined(__CUDACC__)
#define CUGO_HD __host__ __device__
#else
#define CUGO_HD
#endif

namespace cugo::game {

enum class SettlementStatus50 : std::uint8_t {
  kOk = 0,
  kGameNotFinished = 1,
  kInvalidWinner = 2,
};

enum SettlementFlag50 : std::uint16_t {
  kSettlementNone = 0,
  kSettlementFixedSpecial = 1u << 0,
  kSettlementMeongtta = 1u << 1,
  kSettlementPiBak = 1u << 2,
  kSettlementGwangBak = 1u << 3,
  kSettlementDokBak = 1u << 4,
  kSettlementVoid = 1u << 5,
  kSettlementMission = 1u << 6,
};

struct Settlement50 {
  SettlementStatus50 status;
  TerminalReason50 reason;
  std::uint8_t winner;
  std::uint8_t loser;
  std::uint8_t winner_base_points;
  std::uint8_t loser_base_points;
  std::uint8_t winner_pi_units;
  std::uint8_t loser_pi_units;
  std::uint8_t winner_brights;
  std::uint8_t loser_brights;
  std::uint8_t winner_go_count;
  std::uint8_t winner_bombs;
  std::uint8_t winner_shakes;
  std::uint16_t flags;
  std::uint64_t go_multiplier;
  std::uint64_t bomb_shake_multiplier;
  std::uint64_t bak_multiplier;
  std::uint64_t points_after_go;
  std::uint64_t points_after_bomb_shake;
  std::uint64_t final_points;
  std::uint64_t mission_multiplier;
  std::uint64_t round_multiplier;
  // Immediate event payments to player 0 (ppuk/ttadak), outside final_points.
  std::int64_t side_points0;
};

CUGO_HD inline Settlement50 settlement_error50(
    SettlementStatus50 status,
    TerminalReason50 reason = TerminalReason50::kNone) noexcept {
  return Settlement50{status,
                      reason,
                      kNoTerminal50Player,
                      kNoTerminal50Player,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      kSettlementNone,
                      1,
                      1,
                      1,
                      0,
                      0,
                      0,
                      1,
                      1,
                      0};
}

CUGO_HD inline bool settlement_is_fixed_special50(
    TerminalReason50 reason) noexcept {
  return reason == TerminalReason50::kInitialChongtong ||
         reason == TerminalReason50::kFloorChongtong ||
         reason == TerminalReason50::kBonusChongtong ||
         reason == TerminalReason50::kThreePpuk;
}

CUGO_HD inline Settlement50 settle_terminal50(
    const TerminalGameState50& state) noexcept {
  if (!terminal50_is_finished(state)) {
    return settlement_error50(SettlementStatus50::kGameNotFinished);
  }
  if (state.terminal_winner > 1u) {
    return settlement_error50(SettlementStatus50::kInvalidWinner,
                              state.terminal_reason);
  }

  const std::uint8_t winner = state.terminal_winner;
  const std::uint8_t loser = static_cast<std::uint8_t>(winner ^ 1u);
  const ScoreBreakdown winner_score =
      persistent_score_player50(state.special.game.turn, winner);
  const ScoreBreakdown loser_score =
      persistent_score_player50(state.special.game.turn, loser);
  const std::uint8_t go_count =
      go_count_for_player50(state.special.game, winner);
  const std::uint8_t bombs = bombs50(state.special, winner);
  const std::uint8_t shakes = shake_count50(state.special, winner);
  const std::uint64_t round = round_multiplier50(state);
  const std::int64_t side = state.side_points0;

  if (settlement_is_fixed_special50(state.terminal_reason)) {
    // SingleServer MCKLOLPMBOH: Chongtong 10 and three-ppuk 7, times the nagari carry.
    const std::uint64_t points = state.terminal_points * round;
    return Settlement50{SettlementStatus50::kOk,
                        state.terminal_reason,
                        winner,
                        loser,
                        winner_score.total_points,
                        loser_score.total_points,
                        winner_score.pi_units,
                        loser_score.pi_units,
                        winner_score.bright_count,
                        loser_score.bright_count,
                        go_count,
                        bombs,
                        shakes,
                        kSettlementFixedSpecial,
                        1,
                        1,
                        1,
                        state.terminal_points,
                        state.terminal_points,
                        points,
                        1,
                        round,
                        side};
  }

  // SingleServer MCKLOLPMBOH:
  //   (base + go) * 2^(shakes + bombs) * meongtta * mission * go multiplier
  //   * gwang-bak * pi-bak * go-bak * nagari carry; zero when the loser took nothing.
  const GoAdjustedScore50 go =
      go_adjusted_score50(winner_score.total_points, go_count);
  const std::uint64_t special_multiplier =
      bomb_shake_multiplier50(state.special, winner);
  std::uint16_t flags = kSettlementNone;
  std::uint64_t bak_multiplier = 1;

  if ((winner_score.flags & kScoreFlagMeongtta) != 0) {
    flags |= kSettlementMeongtta;
    bak_multiplier *= 2u;
  }
  if (winner_score.pi_units >= 10u && loser_score.pi_units >= 1u &&
      loser_score.pi_units <= 7u) {
    flags |= kSettlementPiBak;
    bak_multiplier *= 2u;
  }
  if (winner_score.bright_count >= 3u && loser_score.bright_count == 0u) {
    flags |= kSettlementGwangBak;
    bak_multiplier *= 2u;
  }
  if (go_count_for_player50(state.special.game, loser) != 0u) {
    flags |= kSettlementDokBak;
    bak_multiplier *= 2u;
  }
  std::uint64_t mission = mission_multi50(state, winner);
  if (mission > 1u) flags |= kSettlementMission;
  if (mission == 0u) mission = 1u;
  const CardMask loser_captured = loser == 0 ? state.special.game.turn.captured0
                                             : state.special.game.turn.captured1;
  const bool void_game = loser_captured == 0;
  if (void_game) flags |= kSettlementVoid;

  const std::uint64_t after_special =
      go.points_after_go * special_multiplier;
  const std::uint64_t final_points =
      void_game ? 0u : after_special * bak_multiplier * mission * round;
  return Settlement50{SettlementStatus50::kOk,
                      state.terminal_reason,
                      winner,
                      loser,
                      winner_score.total_points,
                      loser_score.total_points,
                      winner_score.pi_units,
                      loser_score.pi_units,
                      winner_score.bright_count,
                      loser_score.bright_count,
                      go_count,
                      bombs,
                      shakes,
                      flags,
                      go.go_multiplier,
                      special_multiplier,
                      bak_multiplier,
                      go.points_after_go,
                      after_special,
                      final_points,
                      mission,
                      round,
                      side};
}

// Zero-sum reward: the settled result plus the immediate event payments.
CUGO_HD inline std::int64_t settlement_reward50(
    const Settlement50& settlement,
    std::uint8_t player) noexcept {
  if (settlement.status != SettlementStatus50::kOk || player > 1u) return 0;
  const std::int64_t points = static_cast<std::int64_t>(settlement.final_points);
  const std::int64_t side = player == 0 ? settlement.side_points0 : -settlement.side_points0;
  return (player == settlement.winner ? points : -points) + side;
}

// Nagari has no settlement, but event payments already made still count.
CUGO_HD inline std::int64_t nagari_reward50(const TerminalGameState50& state,
                                            std::uint8_t player) noexcept {
  const std::int64_t side = state.side_points0;
  return player == 0 ? side : -side;
}

}  // namespace cugo::game

#undef CUGO_HD
