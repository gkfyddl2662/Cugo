#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cugo/game/settlement50.h"

namespace {

using cugo::core::CardMask;
using namespace cugo::game;

__host__ __device__ CardMask first_n(CardMask cards, int count) {
  CardMask out = 0;
  while (count-- > 0 && cards != 0) {
    const auto card = cugo::core::pop_first_card(cards);
    out |= cugo::core::card_bit(card);
  }
  return out;
}

__host__ __device__ void set_captured(TerminalGameState50& state,
                                      std::uint8_t player,
                                      CardMask cards) {
  if ((player & 1u) == 0) state.special.game.turn.captured0 = cards;
  else state.special.game.turn.captured1 = cards;
}

__host__ __device__ TerminalGameState50 make_case(std::uint32_t index) {
  const std::uint32_t mode = index & 15u;
  const std::uint8_t winner = static_cast<std::uint8_t>((index >> 4u) & 1u);
  const std::uint8_t loser = static_cast<std::uint8_t>(winner ^ 1u);
  TerminalGameState50 state{};
  state.terminal_winner = winner;
  state.terminal_reason = mode == 0u ? TerminalReason50::kNone
                                     : TerminalReason50::kStop;

  const CardMask plain_winner = kGodoriMask | kHongdanMask;
  const CardMask plain_loser = first_n(kPlainPiMask, 8);
  set_captured(state, winner, plain_winner);
  set_captured(state, loser, plain_loser);

  if (mode == 2u) {
    if (winner == 0) state.special.game.go_count0 = 3;
    else state.special.game.go_count1 = 3;
  } else if (mode == 3u) {
    if (winner == 0) {
      state.special.bombs0 = 2;
      state.special.shaken0 = 1u;
    } else {
      state.special.bombs1 = 2;
      state.special.shaken1 = 1u;
    }
  } else if (mode == 4u) {
    set_captured(state, winner, first_n(kAnimalMask, 7));
  } else if (mode == 5u) {
    const CardMask w = first_n(kBrightMask, 3) | first_n(kPlainPiMask, 10);
    set_captured(state, winner, w);
    set_captured(state, loser, first_n(kPlainPiMask & ~w, 7));
  } else if (mode == 6u) {
    if (loser == 0) state.special.game.go_count0 = 1;
    else state.special.game.go_count1 = 1;
  } else if (mode == 7u) {
    const CardMask w = first_n(kAnimalMask, 7) |
                       first_n(kBrightMask, 3) |
                       first_n(kPlainPiMask, 10);
    set_captured(state, winner, w);
    set_captured(state, loser, first_n(kPlainPiMask & ~w, 7));
    if (winner == 0) {
      state.special.game.go_count0 = 3;
      state.special.bombs0 = 1;
      state.special.shaken0 = 1u;
      state.special.game.go_count1 = 1;
    } else {
      state.special.game.go_count1 = 3;
      state.special.bombs1 = 1;
      state.special.shaken1 = 1u;
      state.special.game.go_count0 = 1;
    }
  } else if (mode == 8u) {
    state.terminal_reason = TerminalReason50::kInitialChongtong;
    state.terminal_points = kChongtongPoints50;
  } else if (mode == 9u) {
    state.terminal_reason = TerminalReason50::kThreePpuk;
    state.terminal_points = kThreeConsecutivePpukPoints50;
  }
  return state;
}

__host__ __device__ Settlement50 run_case(std::uint32_t index) {
  return settle_terminal50(make_case(index));
}

__global__ void settlement_kernel(Settlement50* out, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = run_case(i);
}

bool equal_settlement(const Settlement50& a, const Settlement50& b) {
  return a.status == b.status && a.reason == b.reason &&
         a.winner == b.winner && a.loser == b.loser &&
         a.winner_base_points == b.winner_base_points &&
         a.loser_base_points == b.loser_base_points &&
         a.winner_pi_units == b.winner_pi_units &&
         a.loser_pi_units == b.loser_pi_units &&
         a.winner_brights == b.winner_brights &&
         a.loser_brights == b.loser_brights &&
         a.winner_go_count == b.winner_go_count &&
         a.winner_bombs == b.winner_bombs &&
         a.winner_shakes == b.winner_shakes &&
         a.flags == b.flags && a.go_multiplier == b.go_multiplier &&
         a.bomb_shake_multiplier == b.bomb_shake_multiplier &&
         a.bak_multiplier == b.bak_multiplier &&
         a.points_after_go == b.points_after_go &&
         a.points_after_bomb_shake == b.points_after_bomb_shake &&
         a.final_points == b.final_points;
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(1);
  }
}

}  // namespace

int main() {
  constexpr std::uint32_t kSamples = 65536;
  Settlement50* device = nullptr;
  check_cuda(cudaMalloc(&device, sizeof(Settlement50) * kSamples), "cudaMalloc");

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kSamples + kThreads - 1) / kThreads);
  settlement_kernel<<<blocks, kThreads>>>(device, kSamples);
  check_cuda(cudaGetLastError(), "settlement_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "settlement_kernel sync");

  Settlement50* host = new Settlement50[kSamples];
  check_cuda(cudaMemcpy(host, device, sizeof(Settlement50) * kSamples,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy");

  std::uint32_t stop = 0;
  std::uint32_t fixed = 0;
  std::uint32_t unfinished = 0;
  std::uint32_t all_multipliers = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const Settlement50 expected = run_case(i);
    if (!equal_settlement(host[i], expected)) {
      std::cerr << "cugo_cuda_settlement50_test: FAIL at sample " << i << '\n';
      delete[] host;
      cudaFree(device);
      return 1;
    }
    const std::uint32_t mode = i & 15u;
    if (mode == 0u) ++unfinished;
    else if (mode == 8u || mode == 9u) ++fixed;
    else {
      ++stop;
      if (mode == 7u) ++all_multipliers;
    }
  }

  delete[] host;
  check_cuda(cudaFree(device), "cudaFree");

  std::cout << "cugo_cuda_settlement50_test: PASS (" << kSamples
            << " differential samples; stop=" << stop
            << " fixed=" << fixed
            << " unfinished=" << unfinished
            << " all_multipliers=" << all_multipliers << ")\n";
  return 0;
}
