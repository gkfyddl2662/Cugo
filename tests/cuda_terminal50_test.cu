#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cugo/game/terminal50.h"

namespace {

using namespace cugo::game;
using cugo::core::CardMask;

struct CaseOut {
  std::uint8_t reason;
  std::uint8_t winner;
  std::uint8_t ppuk_count;
  std::uint8_t ppuk_streak;
  std::uint8_t pending_actor;
  std::uint16_t pending_months;
  std::uint64_t points;
};

__host__ __device__ TerminalGameState50 blank_terminal_state() {
  TerminalGameState50 s{};
  s.special.game.decision_actor = kNoGame50Player;
  s.special.game.winner = kNoGame50Player;
  s.special.game.turn.pending_played = cugo::core::kInvalidCard;
  s.special.game.turn.pending_drawn = cugo::core::kInvalidCard;
  s.pending_chongtong_actor = kNoTerminal50Player;
  s.terminal_winner = kNoTerminal50Player;
  s.terminal_reason = TerminalReason50::kNone;
  return s;
}

__host__ __device__ CaseOut run_case(std::uint32_t index) {
  const std::uint32_t mode = index & 7u;
  TerminalGameState50 s = blank_terminal_state();

  if (mode <= 4u) {
    InitialDeal50 d{};
    std::uint8_t first = 0;
    if (mode == 0u) {
      d.hand0 = cugo::core::month_mask(0);
    } else if (mode == 1u) {
      d.hand1 = cugo::core::month_mask(1);
    } else if (mode == 2u) {
      d.hand0 = cugo::core::month_mask(0);
      d.hand1 = cugo::core::month_mask(1);
    } else if (mode == 3u) {
      d.hand0 = cugo::core::month_mask(0);
      d.hand1 = cugo::core::month_mask(1);
      first = 1;
    } else {
      d.floor = cugo::core::month_mask(2);
    }
    const InitialChongtong50 initial = detect_initial_chongtong50(d, first);
    if (initial.reason != TerminalReason50::kNone) {
      mark_terminal50(s, initial.winner, initial.reason, kChongtongPoints50);
    }
  } else if (mode == 5u) {
    open_bonus_chongtong_choice50(s, 0, std::uint16_t{1} << 3);
    apply_chongtong_action50(s, ChongtongAction50::kWin);
  } else if (mode == 6u) {
    record_completed_turn_ppuk50(s, 0, true);
    record_completed_turn_ppuk50(s, 0, false);
    record_completed_turn_ppuk50(s, 0, true);
    record_completed_turn_ppuk50(s, 0, false);
    record_completed_turn_ppuk50(s, 0, true);
  } else {
    record_completed_turn_ppuk50(s, 1, true);
    record_completed_turn_ppuk50(s, 1, true);
    record_completed_turn_ppuk50(s, 1, true);
  }

  const std::uint8_t player = mode == 7u ? 1u : 0u;
  return CaseOut{static_cast<std::uint8_t>(s.terminal_reason),
                 s.terminal_winner,
                 ppuk_count50(s, player),
                 ppuk_streak50(s, player),
                 s.pending_chongtong_actor,
                 s.pending_chongtong_months,
                 s.terminal_points};
}

__global__ void terminal_kernel(CaseOut* out, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = run_case(i);
}

bool equal_case(const CaseOut& a, const CaseOut& b) {
  return a.reason == b.reason && a.winner == b.winner &&
         a.ppuk_count == b.ppuk_count && a.ppuk_streak == b.ppuk_streak &&
         a.pending_actor == b.pending_actor &&
         a.pending_months == b.pending_months && a.points == b.points;
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
  CaseOut* device = nullptr;
  check_cuda(cudaMalloc(&device, sizeof(CaseOut) * kSamples), "cudaMalloc");

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kSamples + kThreads - 1) / kThreads);
  terminal_kernel<<<blocks, kThreads>>>(device, kSamples);
  check_cuda(cudaGetLastError(), "terminal_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "terminal_kernel sync");

  CaseOut* host = new CaseOut[kSamples];
  check_cuda(cudaMemcpy(host, device, sizeof(CaseOut) * kSamples,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy");

  std::uint32_t initial_chongtong = 0;
  std::uint32_t bonus_choice = 0;
  std::uint32_t three_ppuk = 0;
  std::uint32_t consecutive_three_ppuk = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const CaseOut expected = run_case(i);
    if (!equal_case(host[i], expected)) {
      std::cerr << "cugo_cuda_terminal50_test: FAIL at sample " << i << '\n';
      delete[] host;
      cudaFree(device);
      return 1;
    }
    const std::uint32_t mode = i & 7u;
    if (mode <= 4u) ++initial_chongtong;
    else if (mode == 5u) ++bonus_choice;
    else {
      ++three_ppuk;
      if (mode == 7u) ++consecutive_three_ppuk;
    }
  }

  delete[] host;
  check_cuda(cudaFree(device), "cudaFree");

  std::cout << "cugo_cuda_terminal50_test: PASS (" << kSamples
            << " differential samples; initial_chongtong=" << initial_chongtong
            << " bonus_choice=" << bonus_choice
            << " three_ppuk=" << three_ppuk
            << " consecutive_three_ppuk=" << consecutive_three_ppuk << ")\n";
  return 0;
}
