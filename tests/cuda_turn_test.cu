#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/deal.h"
#include "cugo/game/turn.h"

namespace {

using cugo::game::ResolveResult;
using cugo::game::ResolveStatus;
using cugo::game::TurnState48;
using cugo::game::TurnStateSoA48;

constexpr int kThreads = 256;
constexpr int kU64Fields = 7;
constexpr int kU16Fields = 3;
constexpr int kU8Fields = 4;

struct ResolveSnapshot {
  std::uint64_t captured_cards;
  std::uint8_t status;
  std::uint8_t events;
  std::uint8_t captured_own_ppuk;
  std::uint8_t captured_opponent_ppuk;
};

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
  const cugo::core::CardId card =
      cugo::core::first_card(cugo::game::active_hand(state));
  if (card == cugo::core::kInvalidCard ||
      cugo::game::begin_regular_play(state, card) !=
          cugo::game::TurnStatus::kOk ||
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

__global__ void resolve_phase_kernel(TurnStateSoA48 states,
                                     ResolveSnapshot* results,
                                     std::uint8_t* errors,
                                     int games) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) {
    return;
  }

  auto state = cugo::game::load_turn_state(states, static_cast<std::size_t>(game));
  const ResolveResult result = cugo::game::resolve_turn(state);
  if (result.status == ResolveStatus::kOk) {
    if (!cugo::game::is_valid_turn_state(state) ||
        state.phase != cugo::game::TurnPhase::kPlay) {
      errors[game] = 4;
    }
  } else if (result.status == ResolveStatus::kChoiceRequired) {
    if (!cugo::game::is_valid_turn_state(state) ||
        state.phase != cugo::game::TurnPhase::kResolve) {
      errors[game] = 5;
    }
  } else {
    errors[game] = 6;
  }

  results[game] = ResolveSnapshot{
      result.captured_cards,
      static_cast<std::uint8_t>(result.status),
      result.events,
      result.captured_own_ppuk,
      result.captured_opponent_ppuk,
  };
  cugo::game::store_turn_state(states, static_cast<std::size_t>(game), state);
}

__global__ void resolve_benchmark_kernel(TurnStateSoA48 states,
                                         std::uint8_t* statuses,
                                         int games) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) {
    return;
  }

  auto state = cugo::game::load_turn_state(states, static_cast<std::size_t>(game));
  const ResolveResult result = cugo::game::resolve_turn(state);
  statuses[game] = static_cast<std::uint8_t>(result.status);
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

bool same_result(const ResolveSnapshot& gpu, const ResolveResult& cpu) {
  return gpu.captured_cards == cpu.captured_cards &&
         gpu.status == static_cast<std::uint8_t>(cpu.status) &&
         gpu.events == cpu.events &&
         gpu.captured_own_ppuk == cpu.captured_own_ppuk &&
         gpu.captured_opponent_ppuk == cpu.captured_opponent_ppuk;
}

void print_state(const char* name, const TurnState48& state) {
  std::cerr << name << " hand0=0x" << std::hex << state.hand0
            << " hand1=0x" << state.hand1 << " floor=0x" << state.floor
            << " stock=0x" << state.stock << " captured0=0x" << state.captured0
            << " captured1=0x" << state.captured1 << " rng=0x" << state.rng_state
            << std::dec << " ppuk=" << state.ppuk_months
            << " ppuk_owner1=" << state.ppuk_owner1_months
            << " turn=" << state.turn_index
            << " actor=" << static_cast<unsigned>(state.actor)
            << " phase=" << static_cast<unsigned>(state.phase)
            << " played=" << static_cast<unsigned>(state.pending_played)
            << " drawn=" << static_cast<unsigned>(state.pending_drawn) << '\n';
}

int run_differential() {
  constexpr int kGames = 1 << 16;
  constexpr std::uint64_t kMasterSeed = 0x7475726e5f677075ULL;
  const int blocks = (kGames + kThreads - 1) / kThreads;
  const std::size_t games = static_cast<std::size_t>(kGames);

  std::uint64_t* device_u64 = nullptr;
  std::uint16_t* device_u16 = nullptr;
  std::uint8_t* device_u8 = nullptr;
  std::uint8_t* device_errors = nullptr;
  ResolveSnapshot* device_results = nullptr;

  auto cleanup = [&]() {
    cudaFree(device_u64);
    cudaFree(device_u16);
    cudaFree(device_u8);
    cudaFree(device_errors);
    cudaFree(device_results);
  };

  if (!check_cuda(cudaMalloc(&device_u64,
                             games * kU64Fields * sizeof(std::uint64_t)),
                  "cudaMalloc(u64)") ||
      !check_cuda(cudaMalloc(&device_u16,
                             games * kU16Fields * sizeof(std::uint16_t)),
                  "cudaMalloc(u16)") ||
      !check_cuda(cudaMalloc(&device_u8,
                             games * kU8Fields * sizeof(std::uint8_t)),
                  "cudaMalloc(u8)") ||
      !check_cuda(cudaMalloc(&device_errors, games * sizeof(std::uint8_t)),
                  "cudaMalloc(errors)") ||
      !check_cuda(cudaMalloc(&device_results,
                             games * sizeof(ResolveSnapshot)),
                  "cudaMalloc(results)") ||
      !check_cuda(cudaMemset(device_errors, 0, games * sizeof(std::uint8_t)),
                  "cudaMemset(errors)")) {
    cleanup();
    return 1;
  }

  const TurnStateSoA48 device_states =
      make_view(device_u64, device_u16, device_u8, games);
  init_turn_kernel<<<blocks, kThreads>>>(
      device_states, device_errors, kGames, kMasterSeed);
  regular_play_kernel<<<blocks, kThreads>>>(device_states, device_errors, kGames);
  draw_phase_kernel<<<blocks, kThreads>>>(device_states, device_errors, kGames);
  resolve_phase_kernel<<<blocks, kThreads>>>(
      device_states, device_results, device_errors, kGames);

  if (!check_cuda(cudaGetLastError(), "turn phase launch") ||
      !check_cuda(cudaDeviceSynchronize(), "turn phase synchronize")) {
    cleanup();
    return 1;
  }

  std::vector<std::uint64_t> host_u64(games * kU64Fields);
  std::vector<std::uint16_t> host_u16(games * kU16Fields);
  std::vector<std::uint8_t> host_u8(games * kU8Fields);
  std::vector<std::uint8_t> host_errors(games);
  std::vector<ResolveSnapshot> host_results(games);

  if (!check_cuda(cudaMemcpy(host_u64.data(),
                             device_u64,
                             host_u64.size() * sizeof(std::uint64_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(u64)") ||
      !check_cuda(cudaMemcpy(host_u16.data(),
                             device_u16,
                             host_u16.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(u16)") ||
      !check_cuda(cudaMemcpy(host_u8.data(),
                             device_u8,
                             host_u8.size() * sizeof(std::uint8_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(u8)") ||
      !check_cuda(cudaMemcpy(host_errors.data(),
                             device_errors,
                             host_errors.size() * sizeof(std::uint8_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(errors)") ||
      !check_cuda(cudaMemcpy(host_results.data(),
                             device_results,
                             host_results.size() * sizeof(ResolveSnapshot),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(results)")) {
    cleanup();
    return 1;
  }
  cleanup();

  const TurnStateSoA48 gpu_view =
      make_view(host_u64.data(), host_u16.data(), host_u8.data(), games);

  int resolved = 0;
  int choice_required = 0;
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
    if (cugo::game::begin_regular_play(cpu, played) !=
            cugo::game::TurnStatus::kOk ||
        cugo::game::draw_for_turn(cpu) != cugo::game::TurnStatus::kOk) {
      std::cerr << "CPU pre-resolve transition failure at game " << game << '\n';
      return 3;
    }

    const ResolveResult cpu_result = cugo::game::resolve_turn(cpu);
    if (cpu_result.status == ResolveStatus::kOk) {
      ++resolved;
    } else if (cpu_result.status == ResolveStatus::kChoiceRequired) {
      ++choice_required;
    } else {
      std::cerr << "Unexpected CPU resolve status at game " << game
                << " status=" << static_cast<unsigned>(cpu_result.status) << '\n';
      return 4;
    }

    const TurnState48 gpu =
        cugo::game::load_turn_state(gpu_view, static_cast<std::size_t>(game));
    if (!same_state(gpu, cpu) ||
        !same_result(host_results[game], cpu_result) ||
        !cugo::game::is_valid_turn_state(gpu)) {
      std::cerr << "CPU/GPU turn resolve mismatch at game " << game << '\n';
      print_state("gpu", gpu);
      print_state("cpu", cpu);
      return 5;
    }
  }

  if (resolved == 0 || choice_required == 0) {
    std::cerr << "Resolve coverage missing: resolved=" << resolved
              << " choice_required=" << choice_required << '\n';
    return 6;
  }

  std::cout << "cugo_cuda_turn_test: PASS (" << kGames
            << " games PLAY -> DRAW -> RESOLVE; resolved=" << resolved
            << " choice_required=" << choice_required << ")\n";
  return 0;
}

int run_benchmark() {
  constexpr int kGames = 1 << 20;
  constexpr int kWarmups = 8;
  constexpr int kIterations = 256;
  constexpr std::uint64_t kMasterSeed = 0x7265736f6c76655fULL;
  constexpr int kBlockSizes[] = {128, 256, 512};
  const std::size_t games = static_cast<std::size_t>(kGames);

  int device_id = 0;
  cudaDeviceProp properties{};
  cudaFuncAttributes attributes{};
  if (!check_cuda(cudaGetDevice(&device_id), "cudaGetDevice") ||
      !check_cuda(cudaGetDeviceProperties(&properties, device_id),
                  "cudaGetDeviceProperties") ||
      !check_cuda(cudaFuncGetAttributes(&attributes, resolve_benchmark_kernel),
                  "cudaFuncGetAttributes(resolve_benchmark_kernel)")) {
    return 1;
  }

  std::uint64_t* device_u64 = nullptr;
  std::uint16_t* device_u16 = nullptr;
  std::uint8_t* device_u8 = nullptr;
  std::uint8_t* device_errors = nullptr;
  std::uint8_t* device_statuses = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  auto cleanup = [&]() {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    cudaFree(device_u64);
    cudaFree(device_u16);
    cudaFree(device_u8);
    cudaFree(device_errors);
    cudaFree(device_statuses);
  };

  if (!check_cuda(cudaMalloc(&device_u64,
                             games * kU64Fields * sizeof(std::uint64_t)),
                  "cudaMalloc(u64)") ||
      !check_cuda(cudaMalloc(&device_u16,
                             games * kU16Fields * sizeof(std::uint16_t)),
                  "cudaMalloc(u16)") ||
      !check_cuda(cudaMalloc(&device_u8,
                             games * kU8Fields * sizeof(std::uint8_t)),
                  "cudaMalloc(u8)") ||
      !check_cuda(cudaMalloc(&device_errors, games * sizeof(std::uint8_t)),
                  "cudaMalloc(errors)") ||
      !check_cuda(cudaMalloc(&device_statuses, games * sizeof(std::uint8_t)),
                  "cudaMalloc(statuses)") ||
      !check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") ||
      !check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
    cleanup();
    return 1;
  }

  const TurnStateSoA48 device_states =
      make_view(device_u64, device_u16, device_u8, games);
  std::vector<std::uint8_t> host_errors(games);
  std::vector<std::uint8_t> host_statuses(games);

  std::cout << "device=" << properties.name
            << " sms=" << properties.multiProcessorCount
            << " max_threads_per_sm=" << properties.maxThreadsPerMultiProcessor
            << '\n';
  std::cout << "resolve_kernel registers/thread=" << attributes.numRegs
            << " local_bytes/thread=" << attributes.localSizeBytes
            << " static_shared_bytes/block=" << attributes.sharedSizeBytes << '\n';
  std::cout << "benchmark_games=" << kGames
            << " benchmark_warmups=" << kWarmups
            << " benchmark_iterations=" << kIterations << '\n';

  for (const int threads : kBlockSizes) {
    const int blocks = (kGames + threads - 1) / threads;
    if (!check_cuda(cudaMemset(device_errors, 0, games * sizeof(std::uint8_t)),
                    "cudaMemset(errors)")) {
      cleanup();
      return 1;
    }

    for (int warmup = 0; warmup < kWarmups; ++warmup) {
      const std::uint64_t seed =
          kMasterSeed + static_cast<std::uint64_t>(warmup) *
                            0x9e3779b97f4a7c15ULL;
      init_turn_kernel<<<blocks, threads>>>(
          device_states, device_errors, kGames, seed);
      regular_play_kernel<<<blocks, threads>>>(device_states, device_errors, kGames);
      draw_phase_kernel<<<blocks, threads>>>(device_states, device_errors, kGames);
      resolve_benchmark_kernel<<<blocks, threads>>>(
          device_states, device_statuses, kGames);
    }
    if (!check_cuda(cudaGetLastError(), "resolve benchmark warmup launch") ||
        !check_cuda(cudaDeviceSynchronize(),
                    "resolve benchmark warmup synchronize")) {
      cleanup();
      return 1;
    }

    double elapsed_ms = 0.0;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      const std::uint64_t seed =
          kMasterSeed +
          static_cast<std::uint64_t>(iteration + kWarmups) *
              0x9e3779b97f4a7c15ULL;
      init_turn_kernel<<<blocks, threads>>>(
          device_states, device_errors, kGames, seed);
      regular_play_kernel<<<blocks, threads>>>(device_states, device_errors, kGames);
      draw_phase_kernel<<<blocks, threads>>>(device_states, device_errors, kGames);

      if (!check_cuda(cudaEventRecord(start), "cudaEventRecord(start)")) {
        cleanup();
        return 1;
      }
      resolve_benchmark_kernel<<<blocks, threads>>>(
          device_states, device_statuses, kGames);
      if (!check_cuda(cudaGetLastError(), "resolve benchmark launch") ||
          !check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)") ||
          !check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)")) {
        cleanup();
        return 1;
      }

      float iteration_ms = 0.0f;
      if (!check_cuda(cudaEventElapsedTime(&iteration_ms, start, stop),
                      "cudaEventElapsedTime")) {
        cleanup();
        return 1;
      }
      elapsed_ms += iteration_ms;
    }

    if (!check_cuda(cudaMemcpy(host_errors.data(),
                               device_errors,
                               games * sizeof(std::uint8_t),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy(errors)") ||
        !check_cuda(cudaMemcpy(host_statuses.data(),
                               device_statuses,
                               games * sizeof(std::uint8_t),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy(statuses)")) {
      cleanup();
      return 1;
    }

    std::size_t resolved = 0;
    std::size_t choice_required = 0;
    std::size_t other = 0;
    for (std::size_t game = 0; game < games; ++game) {
      if (host_errors[game] != 0) {
        std::cerr << "Benchmark preparation error at game " << game
                  << " code=" << static_cast<unsigned>(host_errors[game]) << '\n';
        cleanup();
        return 2;
      }

      const auto status = static_cast<ResolveStatus>(host_statuses[game]);
      if (status == ResolveStatus::kOk) {
        ++resolved;
      } else if (status == ResolveStatus::kChoiceRequired) {
        ++choice_required;
      } else {
        ++other;
      }
    }

    if (resolved == 0 || choice_required == 0 || other != 0) {
      std::cerr << "Benchmark status coverage invalid: resolved=" << resolved
                << " choice_required=" << choice_required
                << " other=" << other << '\n';
      cleanup();
      return 3;
    }

    int active_blocks_per_sm = 0;
    if (!check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &active_blocks_per_sm,
                        resolve_benchmark_kernel,
                        threads,
                        0),
                    "cudaOccupancyMaxActiveBlocksPerMultiprocessor")) {
      cleanup();
      return 1;
    }

    const double seconds = elapsed_ms / 1000.0;
    const double total_games = static_cast<double>(kGames) * kIterations;
    const double games_per_second = total_games / seconds;
    const double average_kernel_us = elapsed_ms * 1000.0 / kIterations;
    const double occupancy =
        static_cast<double>(active_blocks_per_sm * threads) /
        static_cast<double>(properties.maxThreadsPerMultiProcessor);

    std::cout << std::fixed << std::setprecision(2)
              << "threads=" << threads
              << " elapsed_ms=" << elapsed_ms
              << " avg_kernel_us=" << average_kernel_us
              << " games/s=" << games_per_second
              << " theoretical_occupancy=" << occupancy * 100.0 << "%"
              << " resolved=" << resolved
              << " choice_required=" << choice_required
              << " other=" << other << '\n';
  }

  cleanup();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--benchmark") {
    return run_benchmark();
  }
  if (argc != 1) {
    std::cerr << "usage: cugo_cuda_turn_test [--benchmark]\n";
    return 64;
  }
  return run_differential();
}
