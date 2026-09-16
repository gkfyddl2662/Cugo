#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/game/game50.h"
#include "cugo/game/hwatu.h"

namespace {

using cugo::core::CardMask;
using cugo::game::GameState50;
using cugo::game::GoStopAction50;

struct GameSample {
  CardMask captured0;
  CardMask captured1;
  std::uint64_t adjusted_points;
  std::uint8_t player;
  std::uint8_t mode;
  std::uint8_t base_score;
  std::uint8_t opened;
  std::uint8_t action_status;
  std::uint8_t go_count0;
  std::uint8_t go_count1;
  std::uint8_t last_go0;
  std::uint8_t last_go1;
  std::uint8_t decision_actor;
  std::uint8_t winner;
  std::uint8_t valid;
};

__host__ __device__ CardMask score8_cards() {
  return cugo::game::kGodoriMask | cugo::game::kHongdanMask;
}

__host__ __device__ CardMask score9_cards() {
  return score8_cards() |
         cugo::core::card_bit(cugo::core::make_card(3, 1)) |
         cugo::core::card_bit(cugo::core::make_card(4, 1));
}

__host__ __device__ GameState50 make_sample_state(std::uint32_t i) {
  using namespace cugo::core;
  using namespace cugo::game;
  const std::uint8_t player = static_cast<std::uint8_t>(i & 1u);
  const std::uint8_t mode = static_cast<std::uint8_t>((i >> 1u) & 3u);
  const CardMask captured = mode == 2 ? score9_cards() : score8_cards();

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

  if (mode == 1 || mode == 2) {
    if (player == 0) {
      state.go_count0 = 1;
      state.last_go_base_score0 = 8;
    } else {
      state.go_count1 = 1;
      state.last_go_base_score1 = 8;
    }
  }
  return state;
}

__host__ __device__ GameSample evaluate_sample(std::uint32_t i) {
  using namespace cugo::game;
  GameState50 state = make_sample_state(i);
  const std::uint8_t player = static_cast<std::uint8_t>(i & 1u);
  const std::uint8_t mode = static_cast<std::uint8_t>((i >> 1u) & 3u);
  const std::uint8_t base_score = game50_base_score(state, player);
  const bool opened = maybe_open_go_stop_decision50(state, player);

  std::uint8_t action_status = 0xffu;
  if (opened) {
    const GoStopAction50 action = mode == 3 ? GoStopAction50::kStop
                                            : GoStopAction50::kGo;
    action_status = static_cast<std::uint8_t>(
        apply_go_stop_decision50(state, action));
  }

  const auto adjusted = go_adjusted_score_player50(state, player);
  return GameSample{
      state.turn.captured0,
      state.turn.captured1,
      adjusted.points_after_go,
      player,
      mode,
      base_score,
      static_cast<std::uint8_t>(opened ? 1u : 0u),
      action_status,
      state.go_count0,
      state.go_count1,
      state.last_go_base_score0,
      state.last_go_base_score1,
      state.decision_actor,
      state.winner,
      static_cast<std::uint8_t>(is_valid_game_state50(state) ? 1u : 0u),
  };
}

__global__ void evaluate_game50(GameSample* samples, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  samples[i] = evaluate_sample(i);
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return true;
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool same_sample(const GameSample& a, const GameSample& b) {
  return a.captured0 == b.captured0 && a.captured1 == b.captured1 &&
         a.adjusted_points == b.adjusted_points && a.player == b.player &&
         a.mode == b.mode && a.base_score == b.base_score &&
         a.opened == b.opened && a.action_status == b.action_status &&
         a.go_count0 == b.go_count0 && a.go_count1 == b.go_count1 &&
         a.last_go0 == b.last_go0 && a.last_go1 == b.last_go1 &&
         a.decision_actor == b.decision_actor && a.winner == b.winner &&
         a.valid == b.valid;
}

}  // namespace

int main() {
  constexpr std::uint32_t kSamples = 1u << 16;
  constexpr int kThreads = 256;

  GameSample* device = nullptr;
  if (!check_cuda(cudaMalloc(&device, sizeof(GameSample) * kSamples),
                  "cudaMalloc")) {
    return 1;
  }

  evaluate_game50<<<(kSamples + kThreads - 1) / kThreads, kThreads>>>(
      device, kSamples);
  if (!check_cuda(cudaGetLastError(), "game50 kernel launch") ||
      !check_cuda(cudaDeviceSynchronize(), "game50 kernel synchronize")) {
    cudaFree(device);
    return 1;
  }

  std::vector<GameSample> gpu(kSamples);
  if (!check_cuda(cudaMemcpy(gpu.data(), device,
                             sizeof(GameSample) * kSamples,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy")) {
    cudaFree(device);
    return 1;
  }
  cudaFree(device);

  std::uint32_t opened = 0;
  std::uint32_t no_decision = 0;
  std::uint32_t go_actions = 0;
  std::uint32_t stop_actions = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const GameSample cpu = evaluate_sample(i);
    if (!same_sample(gpu[i], cpu)) {
      std::cerr << "CPU/GPU game50 mismatch at sample " << i << '\n';
      return 2;
    }
    if (cpu.valid == 0) {
      std::cerr << "invalid game50 state at sample " << i << '\n';
      return 3;
    }
    if (cpu.opened != 0) {
      ++opened;
      if (cpu.mode == 3) ++stop_actions;
      else ++go_actions;
    } else {
      ++no_decision;
    }
  }

  if (opened != 49152u || no_decision != 16384u ||
      go_actions != 32768u || stop_actions != 16384u) {
    std::cerr << "game50 coverage mismatch opened=" << opened
              << " no_decision=" << no_decision
              << " go=" << go_actions
              << " stop=" << stop_actions << '\n';
    return 4;
  }

  std::cout << "cugo_cuda_game50_test: PASS (" << kSamples
            << " decision differential samples; opened=" << opened
            << " no_decision=" << no_decision
            << " go=" << go_actions
            << " stop=" << stop_actions << ")\n";
  return 0;
}
