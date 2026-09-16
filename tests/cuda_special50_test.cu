#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include "cugo/core/card.h"
#include "cugo/game/special50.h"

namespace {
struct Sample {
  std::uint16_t shake, bomb, grenade;
  std::uint64_t multiplier;
  std::uint8_t shake_count, mode;
};

__host__ __device__ cugo::game::SpecialGameState50 make_state(std::uint32_t i) {
  using namespace cugo::core;
  using namespace cugo::game;
  SpecialGameState50 s{};
  s.game.turn.phase = Turn50Phase::kPlay;
  s.game.turn.actor = static_cast<std::uint8_t>(i & 1u);
  s.game.decision_actor = kNoGame50Player;
  s.game.winner = kNoGame50Player;
  const std::uint8_t m = static_cast<std::uint8_t>((i / 2u) % 12u);
  const CardMask mm = month_mask(m);
  const CardId removed = static_cast<CardId>(m * 4u + 3u);
  const CardMask three = mm & ~card_bit(removed);
  const CardMask two = card_bit(static_cast<CardId>(m * 4u)) |
                       card_bit(static_cast<CardId>(m * 4u + 1u));
  const std::uint8_t mode = static_cast<std::uint8_t>((i / 24u) % 3u);
  CardMask hand = mode == 2 ? two : three;
  CardMask floor = mode == 0 ? 0 : (mode == 1 ? card_bit(removed) : (mm & ~two));
  if (s.game.turn.actor == 0) s.game.turn.hand0 = hand; else s.game.turn.hand1 = hand;
  s.game.turn.floor = floor;
  s.game.turn.stock = kShinMatgoDeckMask & ~(hand | floor);
  s.shaken0 = static_cast<std::uint16_t>((i >> 8u) & kAllMonthBits50);
  s.shaken1 = static_cast<std::uint16_t>((i >> 10u) & kAllMonthBits50);
  s.bombs0 = static_cast<std::uint8_t>((i >> 4u) & 3u);
  s.bombs1 = static_cast<std::uint8_t>((i >> 6u) & 3u);
  return s;
}

__host__ __device__ Sample eval(std::uint32_t i) {
  using namespace cugo::game;
  const auto s = make_state(i);
  const std::uint8_t actor = s.game.turn.actor;
  return Sample{legal_shake_months50(s), legal_bomb_months50(s), legal_grenade_months50(s),
                bomb_shake_multiplier50(s, actor), shake_count50(s, actor),
                static_cast<std::uint8_t>((i / 24u) % 3u)};
}

__global__ void kernel(Sample* out, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = eval(i);
}
bool check(cudaError_t e, const char* w) {
  if (e == cudaSuccess) return true;
  std::cerr << w << ": " << cudaGetErrorString(e) << '\n'; return false;
}
bool same(const Sample& a, const Sample& b) {
  return a.shake == b.shake && a.bomb == b.bomb && a.grenade == b.grenade &&
         a.multiplier == b.multiplier && a.shake_count == b.shake_count && a.mode == b.mode;
}
}  // namespace

int main() {
  constexpr std::uint32_t N = 1u << 16; constexpr int T = 256;
  Sample* d = nullptr;
  if (!check(cudaMalloc(&d, sizeof(Sample) * N), "cudaMalloc")) return 1;
  kernel<<<(N + T - 1) / T, T>>>(d, N);
  if (!check(cudaGetLastError(), "special50 kernel") || !check(cudaDeviceSynchronize(), "special50 sync")) {
    cudaFree(d); return 1;
  }
  std::vector<Sample> gpu(N);
  if (!check(cudaMemcpy(gpu.data(), d, sizeof(Sample) * N, cudaMemcpyDeviceToHost), "cudaMemcpy")) {
    cudaFree(d); return 1;
  }
  cudaFree(d);
  std::uint32_t shake = 0, bomb = 0, grenade = 0;
  for (std::uint32_t i = 0; i < N; ++i) {
    const Sample cpu = eval(i);
    if (!same(cpu, gpu[i])) { std::cerr << "CPU/GPU special50 mismatch " << i << '\n'; return 2; }
    if (cpu.shake) ++shake;
    if (cpu.bomb) ++bomb;
    if (cpu.grenade) ++grenade;
  }
  if (!shake || !bomb || !grenade) return 3;
  std::cout << "cugo_cuda_special50_test: PASS (" << N
            << " differential samples; shake=" << shake << " bomb=" << bomb
            << " grenade=" << grenade << ")\n";
  return 0;
}
