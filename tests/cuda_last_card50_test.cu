#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/game/pi_transfer.h"
#include "cugo/game/turn50.h"

namespace {

using cugo::core::CardId;
using cugo::core::CardMask;
using cugo::game::Resolve50Result;
using cugo::game::TurnState50;

struct Probe {
  TurnState50 state;
  Resolve50Result result;
  std::uint8_t steal;
  std::uint8_t valid;
};

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(1);
  }
}

__host__ __device__ TurnState50 make_case(std::uint32_t i) {
  using namespace cugo::core;
  using namespace cugo::game;

  const bool ppuk_candidate = (i & 1u) != 0;
  const std::uint8_t actor = static_cast<std::uint8_t>((i >> 1u) & 1u);
  const std::uint32_t bonus_mode = (i >> 2u) & 3u;

  CardMask pending_bonus = 0;
  if (bonus_mode == 1u || bonus_mode == 3u) pending_bonus |= card_bit(kBonusTwoPi);
  if (bonus_mode == 2u || bonus_mode == 3u) pending_bonus |= card_bit(kBonusThreePi);

  const CardId played = ppuk_candidate ? CardId{1} : CardId{0};
  const CardId drawn = ppuk_candidate ? CardId{2} : CardId{1};
  const CardMask floor = ppuk_candidate
      ? (card_bit(CardId{0}) | card_bit(CardId{4}))
      : card_bit(CardId{4});
  const CardMask pending = card_bit(played) | card_bit(drawn) | pending_bonus;
  const CardMask remaining = kShinMatgoDeckMask & ~(floor | pending);

  TurnState50 state{};
  state.floor = floor;
  state.stock = 0;
  if (actor == 0) state.captured1 = remaining;
  else state.captured0 = remaining;
  state.rng_state = 0x6c6173745f637564ULL ^ static_cast<std::uint64_t>(i);
  state.pending_bonus_mask = pending_bonus;
  state.actor = actor;
  state.phase = Turn50Phase::kResolve;
  state.pending_played = played;
  state.pending_drawn = drawn;
  return state;
}

__host__ __device__ Probe run_case(std::uint32_t i) {
  using namespace cugo::game;
  Probe probe{};
  probe.state = make_case(i);
  probe.result = resolve_turn50(probe.state);
  probe.steal = resolve_pi_steal_card_count(probe.result);
  probe.valid = static_cast<std::uint8_t>(is_valid_turn_state50(probe.state));
  return probe;
}

__global__ void last_card_kernel(Probe* out, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = run_case(i);
}

bool same_probe(const Probe& a, const Probe& b) {
  const auto& x = a.state;
  const auto& y = b.state;
  return a.result.status == b.result.status &&
         a.result.events == b.result.events &&
         a.result.captured_own_ppuk == b.result.captured_own_ppuk &&
         a.result.captured_opponent_ppuk == b.result.captured_opponent_ppuk &&
         a.result.captured_cards == b.result.captured_cards &&
         a.steal == b.steal && a.valid == b.valid &&
         x.hand0 == y.hand0 && x.hand1 == y.hand1 &&
         x.floor == y.floor && x.stock == y.stock &&
         x.captured0 == y.captured0 && x.captured1 == y.captured1 &&
         x.rng_state == y.rng_state &&
         x.pending_bonus_mask == y.pending_bonus_mask &&
         x.ppuk_months == y.ppuk_months &&
         x.ppuk_owner1_months == y.ppuk_owner1_months &&
         x.bonus2_ppuk_months == y.bonus2_ppuk_months &&
         x.bonus3_ppuk_months == y.bonus3_ppuk_months &&
         x.turn_index == y.turn_index && x.actor == y.actor &&
         x.phase == y.phase &&
         x.pending_played == y.pending_played &&
         x.pending_drawn == y.pending_drawn;
}

}  // namespace

int main() {
  using namespace cugo::game;

  constexpr std::uint32_t kSamples = 65536;
  std::vector<Probe> actual(kSamples);
  Probe* device = nullptr;
  cuda_check(cudaMalloc(&device, sizeof(Probe) * kSamples), "cudaMalloc");

  constexpr std::uint32_t kThreads = 256;
  const std::uint32_t blocks = (kSamples + kThreads - 1) / kThreads;
  last_card_kernel<<<blocks, kThreads>>>(device, kSamples);
  cuda_check(cudaGetLastError(), "last_card_kernel launch");
  cuda_check(cudaDeviceSynchronize(), "last_card_kernel sync");
  cuda_check(cudaMemcpy(actual.data(), device, sizeof(Probe) * kSamples,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy");
  cuda_check(cudaFree(device), "cudaFree");

  std::uint32_t last_jjok = 0;
  std::uint32_t last_ppuk_candidate = 0;
  std::uint32_t with_bonus = 0;

  for (std::uint32_t i = 0; i < kSamples; ++i) {
    const Probe expected = run_case(i);
    if (!same_probe(expected, actual[i])) {
      std::cerr << "cugo_cuda_last_card50_test: mismatch at sample " << i << '\n';
      return 1;
    }
    if (expected.result.status != Resolve50Status::kOk ||
        expected.result.events != kResolve50EventNone ||
        expected.steal != 0 || expected.valid == 0) {
      std::cerr << "cugo_cuda_last_card50_test: invalid final-card result at sample "
                << i << '\n';
      return 1;
    }

    if ((i & 1u) == 0) ++last_jjok;
    else ++last_ppuk_candidate;
    if (((i >> 2u) & 3u) != 0) ++with_bonus;
  }

  std::cout << "cugo_cuda_last_card50_test: PASS (" << kSamples
            << " final-card differential samples; last_jjok=" << last_jjok
            << " last_ppuk_candidate=" << last_ppuk_candidate
            << " with_bonus=" << with_bonus << ")\n";
  return 0;
}
