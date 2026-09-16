#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "cugo/core/rng.h"
#include "cugo/game/deal.h"

namespace {

using cugo::game::InitialDeal48;

constexpr int kFieldCount = 5;
constexpr int kHand0Field = 0;
constexpr int kHand1Field = 1;
constexpr int kFloorField = 2;
constexpr int kStockField = 3;
constexpr int kRngStateField = 4;

__global__ void deal_kernel(std::uint64_t* fields, int games, std::uint64_t master_seed) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) {
    return;
  }

  const std::uint64_t seed =
      cugo::core::derive_seed(master_seed, static_cast<std::uint64_t>(game));
  const InitialDeal48 deal = cugo::game::deal_base_48(seed);
  const std::size_t stride = static_cast<std::size_t>(games);
  const std::size_t index = static_cast<std::size_t>(game);

  fields[index + stride * kHand0Field] = deal.hand0;
  fields[index + stride * kHand1Field] = deal.hand1;
  fields[index + stride * kFloorField] = deal.floor;
  fields[index + stride * kStockField] = deal.stock;
  fields[index + stride * kRngStateField] = deal.rng_state;
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool compare_field(const std::vector<std::uint64_t>& gpu,
                   int games,
                   int game,
                   int field,
                   std::uint64_t expected,
                   const char* name) {
  const std::size_t index = static_cast<std::size_t>(game) +
                            static_cast<std::size_t>(games) * field;
  if (gpu[index] == expected) {
    return true;
  }

  std::cerr << "CPU/GPU deal mismatch at game " << game << ", field " << name
            << ": gpu=0x" << std::hex << gpu[index] << " cpu=0x" << expected
            << std::dec << '\n';
  return false;
}

int run_differential() {
  constexpr int kGames = 1 << 16;
  constexpr int kThreads = 256;
  constexpr std::uint64_t kMasterSeed = 0x4d6174676f5f4750ULL;
  const std::size_t words = static_cast<std::size_t>(kGames) * kFieldCount;

  std::uint64_t* device = nullptr;
  if (!check_cuda(cudaMalloc(&device, words * sizeof(std::uint64_t)), "cudaMalloc")) {
    return 1;
  }

  deal_kernel<<<(kGames + kThreads - 1) / kThreads, kThreads>>>(device, kGames, kMasterSeed);
  if (!check_cuda(cudaGetLastError(), "deal kernel launch")) {
    cudaFree(device);
    return 1;
  }

  std::vector<std::uint64_t> gpu(words);
  if (!check_cuda(cudaMemcpy(gpu.data(), device, words * sizeof(std::uint64_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy")) {
    cudaFree(device);
    return 1;
  }
  cudaFree(device);

  for (int game = 0; game < kGames; ++game) {
    const std::uint64_t seed =
        cugo::core::derive_seed(kMasterSeed, static_cast<std::uint64_t>(game));
    const InitialDeal48 cpu = cugo::game::deal_base_48(seed);
    if (!cugo::game::is_valid_initial_deal(cpu)) {
      std::cerr << "CPU invariant failure at game " << game << '\n';
      return 2;
    }

    if (!compare_field(gpu, kGames, game, kHand0Field, cpu.hand0, "hand0") ||
        !compare_field(gpu, kGames, game, kHand1Field, cpu.hand1, "hand1") ||
        !compare_field(gpu, kGames, game, kFloorField, cpu.floor, "floor") ||
        !compare_field(gpu, kGames, game, kStockField, cpu.stock, "stock") ||
        !compare_field(gpu, kGames, game, kRngStateField, cpu.rng_state, "rng_state")) {
      return 3;
    }
  }

  std::cout << "cugo_cuda_deal_test: PASS (" << kGames << " CPU/GPU deals)\n";
  return 0;
}

int run_benchmark() {
  constexpr int kGames = 1 << 20;
  constexpr int kWarmups = 3;
  constexpr int kIterations = 12;
  constexpr std::uint64_t kMasterSeed = 0x62656e63685f3031ULL;
  constexpr std::array<int, 3> kBlockSizes{128, 256, 512};
  const std::size_t words = static_cast<std::size_t>(kGames) * kFieldCount;

  int device_id = 0;
  cudaDeviceProp properties{};
  cudaFuncAttributes attributes{};
  if (!check_cuda(cudaGetDevice(&device_id), "cudaGetDevice") ||
      !check_cuda(cudaGetDeviceProperties(&properties, device_id), "cudaGetDeviceProperties") ||
      !check_cuda(cudaFuncGetAttributes(&attributes, deal_kernel), "cudaFuncGetAttributes")) {
    return 1;
  }

  std::uint64_t* device = nullptr;
  if (!check_cuda(cudaMalloc(&device, words * sizeof(std::uint64_t)), "cudaMalloc")) {
    return 1;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") ||
      !check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
    cudaFree(device);
    return 1;
  }

  std::cout << "device=" << properties.name << " sms=" << properties.multiProcessorCount
            << " max_threads_per_sm=" << properties.maxThreadsPerMultiProcessor << '\n';
  std::cout << "deal_kernel registers/thread=" << attributes.numRegs
            << " local_bytes/thread=" << attributes.localSizeBytes
            << " static_shared_bytes/block=" << attributes.sharedSizeBytes << '\n';

  for (const int threads : kBlockSizes) {
    const int blocks = (kGames + threads - 1) / threads;
    int active_blocks_per_sm = 0;
    if (!check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &active_blocks_per_sm, deal_kernel, threads, 0),
                    "cudaOccupancyMaxActiveBlocksPerMultiprocessor")) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(device);
      return 1;
    }

    for (int i = 0; i < kWarmups; ++i) {
      deal_kernel<<<blocks, threads>>>(device, kGames,
                                       kMasterSeed + static_cast<std::uint64_t>(i));
    }
    if (!check_cuda(cudaGetLastError(), "benchmark warmup launch") ||
        !check_cuda(cudaDeviceSynchronize(), "benchmark warmup synchronize")) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(device);
      return 1;
    }

    if (!check_cuda(cudaEventRecord(start), "cudaEventRecord(start)")) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(device);
      return 1;
    }
    for (int i = 0; i < kIterations; ++i) {
      const std::uint64_t seed =
          kMasterSeed + static_cast<std::uint64_t>(i + kWarmups) * 0x9e3779b97f4a7c15ULL;
      deal_kernel<<<blocks, threads>>>(device, kGames, seed);
    }
    if (!check_cuda(cudaGetLastError(), "benchmark launch") ||
        !check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)") ||
        !check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)")) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(device);
      return 1;
    }

    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime")) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(device);
      return 1;
    }

    const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
    const double total_games = static_cast<double>(kGames) * kIterations;
    const double games_per_second = total_games / seconds;
    const double sampled_cards_per_second =
        games_per_second * static_cast<double>(cugo::game::kInitialSampledCards);
    const double occupancy =
        static_cast<double>(active_blocks_per_sm * threads) /
        static_cast<double>(properties.maxThreadsPerMultiProcessor);

    std::cout << std::fixed << std::setprecision(2)
              << "threads=" << threads << " elapsed_ms=" << elapsed_ms
              << " games/s=" << games_per_second
              << " sampled_cards/s=" << sampled_cards_per_second
              << " theoretical_occupancy=" << occupancy * 100.0 << "%\n";
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--benchmark") {
    return run_benchmark();
  }
  if (argc != 1) {
    std::cerr << "usage: cugo_cuda_deal_test [--benchmark]\n";
    return 64;
  }
  return run_differential();
}
