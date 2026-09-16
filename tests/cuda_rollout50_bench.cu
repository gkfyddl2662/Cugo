#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cugo/game/rollout50.h"

namespace {

using namespace cugo::game;

struct BenchOut {
  std::uint64_t points;
  std::uint16_t actions;
  std::uint16_t turns;
  std::uint8_t end;
  std::uint8_t winner;
};

__global__ void rollout_bench_kernel(BenchOut* out,
                                     std::uint32_t count,
                                     std::uint64_t seed_base) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint64_t seed = seed_base +
      static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull;
  const RolloutResult50 result =
      rollout_canonical50(seed, static_cast<std::uint8_t>(i & 1u));
  out[i] = BenchOut{result.final_points, result.actions, result.turns,
                    static_cast<std::uint8_t>(result.end), result.winner};
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(1);
  }
}

void run_block_size(BenchOut* out,
                    std::uint32_t games,
                    int warmups,
                    int iterations,
                    int threads) {
  const int blocks = static_cast<int>((games + threads - 1) / threads);
  for (int i = 0; i < warmups; ++i) {
    rollout_bench_kernel<<<blocks, threads>>>(
        out, games, 0x6a09e667f3bcc909ull + static_cast<std::uint64_t>(i) * games);
  }
  check_cuda(cudaGetLastError(), "warmup launch");
  check_cuda(cudaDeviceSynchronize(), "warmup sync");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  check_cuda(cudaEventRecord(start), "cudaEventRecord start");
  for (int i = 0; i < iterations; ++i) {
    rollout_bench_kernel<<<blocks, threads>>>(
        out, games,
        0xbb67ae8584caa73bull + static_cast<std::uint64_t>(i) * games);
  }
  check_cuda(cudaGetLastError(), "benchmark launch");
  check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
  check_cuda(cudaEventSynchronize(stop), "benchmark sync");

  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
             "cudaEventElapsedTime");
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

  cudaFuncAttributes attr{};
  check_cuda(cudaFuncGetAttributes(&attr, rollout_bench_kernel),
             "cudaFuncGetAttributes");
  int active_blocks = 0;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &active_blocks, rollout_bench_kernel, threads, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

  cudaDeviceProp prop{};
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  const double active_threads =
      static_cast<double>(active_blocks) * static_cast<double>(threads);
  const double occupancy = active_threads /
      static_cast<double>(prop.maxThreadsPerMultiProcessor);
  const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
  const double total_games =
      static_cast<double>(games) * static_cast<double>(iterations);
  const double games_per_second = total_games / seconds;
  const double us_per_game = seconds * 1.0e6 / total_games;

  std::cout << "threads=" << threads
            << " blocks=" << blocks
            << " elapsed_ms=" << std::fixed << std::setprecision(3) << elapsed_ms
            << " games_per_s=" << std::setprecision(3) << games_per_second
            << " us_per_game=" << std::setprecision(6) << us_per_game
            << " regs_per_thread=" << attr.numRegs
            << " local_bytes_per_thread=" << attr.localSizeBytes
            << " active_blocks_per_sm=" << active_blocks
            << " occupancy=" << std::setprecision(2) << occupancy * 100.0 << "%\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t games = 131072;
  int iterations = 32;
  int warmups = 4;
  if (argc > 1) games = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
  if (argc > 2) iterations = std::atoi(argv[2]);
  if (argc > 3) warmups = std::atoi(argv[3]);
  if (games == 0 || iterations <= 0 || warmups < 0) {
    std::cerr << "usage: cugo_cuda_rollout50_bench [games] [iterations] [warmups]\n";
    return 1;
  }

  cudaDeviceProp prop{};
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  std::cout << "device=" << prop.name
            << " sms=" << prop.multiProcessorCount
            << " games=" << games
            << " iterations=" << iterations
            << " warmups=" << warmups << '\n';

  BenchOut* out = nullptr;
  check_cuda(cudaMalloc(&out, sizeof(BenchOut) * games), "cudaMalloc");
  run_block_size(out, games, warmups, iterations, 128);
  run_block_size(out, games, warmups, iterations, 256);
  run_block_size(out, games, warmups, iterations, 512);

  auto* host = new BenchOut[games];
  check_cuda(cudaMemcpy(host, out, sizeof(BenchOut) * games,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy final");
  std::uint32_t terminal = 0;
  std::uint32_t nagari = 0;
  std::uint32_t failures = 0;
  std::uint64_t action_sum = 0;
  std::uint64_t turn_sum = 0;
  for (std::uint32_t i = 0; i < games; ++i) {
    const auto end = static_cast<RolloutEnd50>(host[i].end);
    if (end == RolloutEnd50::kTerminal) ++terminal;
    else if (end == RolloutEnd50::kNagari) ++nagari;
    else ++failures;
    action_sum += host[i].actions;
    turn_sum += host[i].turns;
  }
  std::cout << "last_batch terminal=" << terminal
            << " nagari=" << nagari
            << " failures=" << failures
            << " avg_actions=" << std::fixed << std::setprecision(3)
            << static_cast<double>(action_sum) / games
            << " avg_turns=" << static_cast<double>(turn_sum) / games << '\n';

  delete[] host;
  check_cuda(cudaFree(out), "cudaFree");
  if (failures != 0) return 1;
  return 0;
}
