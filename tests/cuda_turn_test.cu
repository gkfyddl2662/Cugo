#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/deal.h"
#include "cugo/game/turn.h"

namespace {

using cugo::game::TurnState48;
using cugo::game::TurnStateSoA48;

constexpr int kThreads = 256;
constexpr int kU64Fields = 7;
constexpr int kU16Fields = 3;
constexpr int kU8Fields = 4;

TurnStateSoA48 make_view(std::uint64_t* u64,
                         std::uint16_t* u16,
                         std::uint8_t* u8,
                         std::size_t games) {
  return TurnStateSoA48{
      u64 + games * 0,
      u64 + games * 1,
      u64 + games * 2,
      u64 + games * 3,
      u64 + games * 4,
      u64 + games * 5,
      u64 + games * 6,
      u16 + games * 0,
      u16 + games * 1,
      u16 + games * 2,
      u8 + games * 0,
      u8 + games * 1,
      u8 + games * 2,
      u8 + games * 3,
  };
}

__global__ void init_turn_kernel(TurnStateSoA48 states,
                                 std::uint8_t* errors,
                                 int games,
                                 std::uint64_t master_seed) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) {
    return;
  }

  const std::uint64_t seed =
      cugo::core::derive_seed(master_seed, static_cast<std::uint64_t>(game));
  const auto deal = cugo::game::deal_base_48(seed);
  const auto state = cugo::game::make_turn_state(
      deal, static_cast<std::uint8_t>(game & 1));
  if (!cugo::game::is_valid_turn_state(state)) {
    errors[game] = 1;
  }
  cugo::game::store_turn_state(states, static_cast<std::size_t>(game), state);
}

__global__ void regular_play_kernel(TurnStateSoA48 states,
                                    std::uint8_t* errors,
                                    int games) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) {
    return;
  }

  auto state = cugo::game::load_turn_state(states, static_cast<std::size_t>(game));
  const cugo::core::CardId card = cugo::core::first_card(cugo::game::active_hand(state));
  if (card == cugo::core::kInvalidCard ||
      cugo::game::begin_regular_play(state, card) != cugo::game::TurnStatus::kOk ||
      !cugo::game::is_valid_turn_state(state)) {
    errors[game] = 2;
  }
  cugo::game::store_turn_state(states, static_cast<std::size_t>(game), state);
}

__global__ void draw_phase_kernel(TurnStateSoA48 states,
                                  std::uint8_t* errors,
                                  int games) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) {
    return;
  }

  auto state = cugo::game::load_turn_state(states, static_cast<std::size_t>(game));
  if (cugo::game::draw_for_turn(state) != cugo::game::TurnStatus::kOk ||
      !cugo::game::is_valid_turn_state(state)) {
    errors[game] = 3;
  }
  cugo::game::store_turn_state(states, static_cast<std::size_t>(game), state);
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool same_state(const TurnState48& gpu, const TurnState48& cpu) {
  return gpu.hand0 == cpu.hand0 && gpu.hand1 == cpu.hand1 &&
         gpu.floor == cpu.floor && gpu.stock == cpu.stock &&
         gpu.captured0 == cpu.captured0 && gpu.captured1 == cpu.captured1 &&
         gpu.rng_state == cpu.rng_state && gpu.ppuk_months == cpu.ppuk_months &&
         gpu.ppuk_owner1_months == cpu.ppuk_owner1_months &&
         gpu.turn_index == cpu.turn_index && gpu.actor == cpu.actor &&
         gpu.phase == cpu.phase && gpu.pending_played == cpu.pending_played &&
         gpu.pending_drawn == cpu.pending_drawn;
}

void print_state(const char* name, const TurnState48& state) {
  std::cerr << name << " hand0=0x" << std::hex << state.hand0
            << " hand1=0x" << state.hand1 << " floor=0x" << state.floor
            << " stock=0x" << state.stock << " captured0=0x" << state.captured0
            << " captured1=0x" << state.captured1 << " rng=0x" << state.rng_state
            << std::dec << " ppuk=" << state.ppuk_months
            << " ppuk_owner1=" << state.ppuk_owner1_months
            << " turn=" << state.turn_index << " actor=" << static_cast<unsigned>(state.actor)
            << " phase=" << static_cast<unsigned>(state.phase)
            << " played=" << static_cast<unsigned>(state.pending_played)
            << " drawn=" << static_cast<unsigned>(state.pending_drawn) << '\n';
}

}  // namespace

int main() {
  constexpr int kGames = 1 << 16;
  constexpr std::uint64_t kMasterSeed = 0x7475726e5f677075ULL;
  const int blocks = (kGames + kThreads - 1) / kThreads;
  const std::size_t games = static_cast<std::size_t>(kGames);

  std::uint64_t* device_u64 = nullptr;
  std::uint16_t* device_u16 = nullptr;
  std::uint8_t* device_u8 = nullptr;
  std::uint8_t* device_errors = nullptr;

  auto cleanup = [&]() {
    cudaFree(device_u64);
    cudaFree(device_u16);
    cudaFree(device_u8);
    cudaFree(device_errors);
  };

  if (!check_cuda(cudaMalloc(&device_u64, games * kU64Fields * sizeof(std::uint64_t)),
                  "cudaMalloc(u64)") ||
      !check_cuda(cudaMalloc(&device_u16, games * kU16Fields * sizeof(std::uint16_t)),
                  "cudaMalloc(u16)") ||
      !check_cuda(cudaMalloc(&device_u8, games * kU8Fields * sizeof(std::uint8_t)),
                  "cudaMalloc(u8)") ||
      !check_cuda(cudaMalloc(&device_errors, games * sizeof(std::uint8_t)),
                  "cudaMalloc(errors)") ||
      !check_cuda(cudaMemset(device_errors, 0, games * sizeof(std::uint8_t)),
                  "cudaMemset(errors)")) {
    cleanup();
    return 1;
  }

  const TurnStateSoA48 device_states =
      make_view(device_u64, device_u16, device_u8, games);
  init_turn_kernel<<<blocks, kThreads>>>(device_states, device_errors, kGames, kMasterSeed);
  regular_play_kernel<<<blocks, kThreads>>>(device_states, device_errors, kGames);
  draw_phase_kernel<<<blocks, kThreads>>>(device_states, device_errors, kGames);

  if (!check_cuda(cudaGetLastError(), "turn phase launch") ||
      !check_cuda(cudaDeviceSynchronize(), "turn phase synchronize")) {
    cleanup();
    return 1;
  }

  std::vector<std::uint64_t> host_u64(games * kU64Fields);
  std::vector<std::uint16_t> host_u16(games * kU16Fields);
  std::vector<std::uint8_t> host_u8(games * kU8Fields);
  std::vector<std::uint8_t> host_errors(games);

  if (!check_cuda(cudaMemcpy(host_u64.data(), device_u64,
                             host_u64.size() * sizeof(std::uint64_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(u64)") ||
      !check_cuda(cudaMemcpy(host_u16.data(), device_u16,
                             host_u16.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(u16)") ||
      !check_cuda(cudaMemcpy(host_u8.data(), device_u8,
                             host_u8.size() * sizeof(std::uint8_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(u8)") ||
      !check_cuda(cudaMemcpy(host_errors.data(), device_errors,
                             host_errors.size() * sizeof(std::uint8_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(errors)")) {
    cleanup();
    return 1;
  }
  cleanup();

  const TurnStateSoA48 gpu_view =
      make_view(host_u64.data(), host_u16.data(), host_u8.data(), games);

  for (int game = 0; game < kGames; ++game) {
    if (host_errors[game] != 0) {
      std::cerr << "GPU invariant/transition error at game " << game
                << " code=" << static_cast<unsigned>(host_errors[game]) << '\n';
      return 2;
    }

    const std::uint64_t seed =
        cugo::core::derive_seed(kMasterSeed, static_cast<std::uint64_t>(game));
    const auto deal = cugo::game::deal_base_48(seed);
    auto cpu = cugo::game::make_turn_state(
        deal, static_cast<std::uint8_t>(game & 1));
    const cugo::core::CardId played =
        cugo::core::first_card(cugo::game::active_hand(cpu));
    if (cugo::game::begin_regular_play(cpu, played) != cugo::game::TurnStatus::kOk ||
        cugo::game::draw_for_turn(cpu) != cugo::game::TurnStatus::kOk ||
        !cugo::game::is_valid_turn_state(cpu)) {
      std::cerr << "CPU transition failure at game " << game << '\n';
      return 3;
    }

    const TurnState48 gpu =
        cugo::game::load_turn_state(gpu_view, static_cast<std::size_t>(game));
    if (!same_state(gpu, cpu) || !cugo::game::is_valid_turn_state(gpu)) {
      std::cerr << "CPU/GPU turn state mismatch at game " << game << '\n';
      print_state("gpu", gpu);
      print_state("cpu", cpu);
      return 4;
    }
  }

  std::cout << "cugo_cuda_turn_test: PASS (" << kGames
            << " games INIT -> PLAY -> DRAW -> RESOLVE)\n";
  return 0;
}
