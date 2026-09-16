#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/deal.h"
#include "cugo/game/state.h"

namespace {

using cugo::game::ResidentStateSoA48;

constexpr int kStateFieldCount = 5;
constexpr int kThreads = 256;

ResidentStateSoA48 make_view(std::uint64_t* base, std::size_t games) {
  return ResidentStateSoA48{
      base + games * 0,
      base + games * 1,
      base + games * 2,
      base + games * 3,
      base + games * 4,
  };
}

__global__ void init_resident_state_kernel(ResidentStateSoA48 states,
                                           int games,
                                           std::uint64_t master_seed) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) {
    return;
  }

  const std::uint64_t seed =
      cugo::core::derive_seed(master_seed, static_cast<std::uint64_t>(game));
  const auto deal = cugo::game::deal_base_48(seed);
  cugo::game::store_initial_deal(states, static_cast<std::size_t>(game), deal);
}

__global__ void draw_stock_kernel(ResidentStateSoA48 states,
                                  std::uint8_t* drawn,
                                  int games) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) {
    return;
  }

  const std::size_t index = static_cast<std::size_t>(game);
  cugo::core::CardMask stock = states.stock[index];
  std::uint64_t rng_state = states.rng_state[index];
  const cugo::core::CardId card = cugo::game::draw_stock_card(stock, rng_state);
  states.stock[index] = stock;
  states.rng_state[index] = rng_state;
  drawn[index] = card;
}

void enqueue_stock_round(ResidentStateSoA48 states,
                         std::uint8_t* drawn,
                         int games,
                         int blocks,
                         std::uint64_t master_seed,
                         cudaStream_t stream) {
  init_resident_state_kernel<<<blocks, kThreads, 0, stream>>>(
      states, games, master_seed);
  for (int step = 0; step < cugo::game::kBaseStockCards; ++step) {
    draw_stock_kernel<<<blocks, kThreads, 0, stream>>>(states, drawn, games);
  }
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool compare_u64(std::uint64_t actual,
                 std::uint64_t expected,
                 int game,
                 const char* field) {
  if (actual == expected) {
    return true;
  }
  std::cerr << "CPU/GPU state mismatch at game " << game << ", field " << field
            << ": gpu=0x" << std::hex << actual << " cpu=0x" << expected
            << std::dec << '\n';
  return false;
}

int run_differential() {
  constexpr int kGames = 1 << 16;
  constexpr int kSteps = cugo::game::kBaseStockCards;
  constexpr std::uint64_t kMasterSeed = 0x73746174655f3031ULL;
  const int blocks = (kGames + kThreads - 1) / kThreads;
  const std::size_t words = static_cast<std::size_t>(kGames) * kStateFieldCount;
  const std::size_t trace_bytes = static_cast<std::size_t>(kGames) * kSteps;

  std::uint64_t* device_words = nullptr;
  std::uint8_t* device_trace = nullptr;
  if (!check_cuda(cudaMalloc(&device_words, words * sizeof(std::uint64_t)),
                  "cudaMalloc(state)") ||
      !check_cuda(cudaMalloc(&device_trace, trace_bytes), "cudaMalloc(trace)")) {
    cudaFree(device_words);
    cudaFree(device_trace);
    return 1;
  }

  const ResidentStateSoA48 device_states =
      make_view(device_words, static_cast<std::size_t>(kGames));
  init_resident_state_kernel<<<blocks, kThreads>>>(device_states, kGames, kMasterSeed);
  for (int step = 0; step < kSteps; ++step) {
    draw_stock_kernel<<<blocks, kThreads>>>(
        device_states,
        device_trace + static_cast<std::size_t>(step) * kGames,
        kGames);
  }
  if (!check_cuda(cudaGetLastError(), "resident transition launch")) {
    cudaFree(device_words);
    cudaFree(device_trace);
    return 1;
  }

  std::vector<std::uint64_t> host_words(words);
  std::vector<std::uint8_t> host_trace(trace_bytes);
  if (!check_cuda(cudaMemcpy(host_words.data(), device_words,
                             words * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(state)") ||
      !check_cuda(cudaMemcpy(host_trace.data(), device_trace, trace_bytes,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(trace)")) {
    cudaFree(device_words);
    cudaFree(device_trace);
    return 1;
  }
  cudaFree(device_words);
  cudaFree(device_trace);

  const ResidentStateSoA48 gpu =
      make_view(host_words.data(), static_cast<std::size_t>(kGames));
  for (int game = 0; game < kGames; ++game) {
    const std::uint64_t seed =
        cugo::core::derive_seed(kMasterSeed, static_cast<std::uint64_t>(game));
    const auto deal = cugo::game::deal_base_48(seed);
    auto cpu = cugo::game::make_resident_state(deal);

    if (!compare_u64(gpu.hand0[game], cpu.hand0, game, "hand0") ||
        !compare_u64(gpu.hand1[game], cpu.hand1, game, "hand1") ||
        !compare_u64(gpu.floor[game], cpu.floor, game, "floor")) {
      return 2;
    }

    for (int step = 0; step < kSteps; ++step) {
      const auto expected = cugo::game::draw_stock_card(cpu);
      const auto actual =
          host_trace[static_cast<std::size_t>(step) * kGames + game];
      if (actual != expected) {
        std::cerr << "CPU/GPU draw mismatch at game " << game << ", step " << step
                  << ": gpu=" << static_cast<unsigned>(actual)
                  << " cpu=" << static_cast<unsigned>(expected) << '\n';
        return 3;
      }
    }

    if (!compare_u64(gpu.stock[game], cpu.stock, game, "stock") ||
        !compare_u64(gpu.rng_state[game], cpu.rng_state, game, "rng_state")) {
      return 4;
    }
    if (cpu.stock != 0 || cugo::core::card_count(cpu.stock) != 0) {
      std::cerr << "CPU stock did not drain at game " << game << '\n';
      return 5;
    }
  }

  std::cout << "cugo_cuda_state_test: PASS (" << kGames << " games x " << kSteps
            << " resident stock transitions)\n";
  return 0;
}

int run_benchmark() {
  constexpr int kGames = 1 << 20;
  constexpr int kSteps = cugo::game::kBaseStockCards;
  constexpr int kWarmups = 8;
  constexpr int kIterations = 256;
  constexpr std::uint64_t kMasterSeed = 0x73746174655f626dULL;
  const int blocks = (kGames + kThreads - 1) / kThreads;
  const std::size_t words = static_cast<std::size_t>(kGames) * kStateFieldCount;

  int device_id = 0;
  cudaDeviceProp properties{};
  cudaFuncAttributes attributes{};
  if (!check_cuda(cudaGetDevice(&device_id), "cudaGetDevice") ||
      !check_cuda(cudaGetDeviceProperties(&properties, device_id),
                  "cudaGetDeviceProperties") ||
      !check_cuda(cudaFuncGetAttributes(&attributes, draw_stock_kernel),
                  "cudaFuncGetAttributes(draw_stock_kernel)")) {
    return 1;
  }

  int active_blocks_per_sm = 0;
  if (!check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &active_blocks_per_sm, draw_stock_kernel, kThreads, 0),
                  "cudaOccupancyMaxActiveBlocksPerMultiprocessor")) {
    return 1;
  }

  std::uint64_t* device_words = nullptr;
  std::uint8_t* device_drawn = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  auto cleanup = [&]() {
    if (graph_exec != nullptr) {
      cudaGraphExecDestroy(graph_exec);
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    cudaFree(device_words);
    cudaFree(device_drawn);
  };

  if (!check_cuda(cudaMalloc(&device_words, words * sizeof(std::uint64_t)),
                  "cudaMalloc(state)") ||
      !check_cuda(cudaMalloc(&device_drawn, static_cast<std::size_t>(kGames)),
                  "cudaMalloc(drawn)") ||
      !check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags") ||
      !check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") ||
      !check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
    cleanup();
    return 1;
  }

  const ResidentStateSoA48 device_states =
      make_view(device_words, static_cast<std::size_t>(kGames));

  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    enqueue_stock_round(device_states, device_drawn, kGames, blocks,
                        kMasterSeed + static_cast<std::uint64_t>(warmup), stream);
  }
  if (!check_cuda(cudaGetLastError(), "benchmark warmup launch") ||
      !check_cuda(cudaStreamSynchronize(stream), "benchmark warmup synchronize")) {
    cleanup();
    return 1;
  }

  double draw_elapsed_ms = 0.0;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const std::uint64_t master_seed =
        kMasterSeed + static_cast<std::uint64_t>(iteration + kWarmups) *
                          0x9e3779b97f4a7c15ULL;
    init_resident_state_kernel<<<blocks, kThreads, 0, stream>>>(
        device_states, kGames, master_seed);
    if (!check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)")) {
      cleanup();
      return 1;
    }
    for (int step = 0; step < kSteps; ++step) {
      draw_stock_kernel<<<blocks, kThreads, 0, stream>>>(
          device_states, device_drawn, kGames);
    }
    if (!check_cuda(cudaGetLastError(), "benchmark transition launch") ||
        !check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)") ||
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
    draw_elapsed_ms += iteration_ms;
  }

  const double draw_seconds = draw_elapsed_ms / 1000.0;
  const double draw_transitions =
      static_cast<double>(kGames) * kSteps * kIterations;
  const double draw_rounds = static_cast<double>(kGames) * kIterations;
  const double draw_transitions_per_second = draw_transitions / draw_seconds;
  const double draw_rounds_per_second = draw_rounds / draw_seconds;
  const double average_kernel_us =
      draw_elapsed_ms * 1000.0 / (kSteps * kIterations);
  const double occupancy =
      static_cast<double>(active_blocks_per_sm * kThreads) /
      static_cast<double>(properties.maxThreadsPerMultiProcessor);

  const auto graph_setup_start = std::chrono::steady_clock::now();
  if (!check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                  "cudaStreamBeginCapture")) {
    cleanup();
    return 1;
  }
  enqueue_stock_round(device_states, device_drawn, kGames, blocks, kMasterSeed,
                      stream);
  if (!check_cuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture") ||
      !check_cuda(cudaGraphInstantiate(&graph_exec, graph, 0ULL),
                  "cudaGraphInstantiate")) {
    cleanup();
    return 1;
  }
  std::size_t graph_nodes = 0;
  if (!check_cuda(cudaGraphGetNodes(graph, nullptr, &graph_nodes),
                  "cudaGraphGetNodes")) {
    cleanup();
    return 1;
  }
  const auto graph_setup_stop = std::chrono::steady_clock::now();
  const double graph_setup_ms =
      std::chrono::duration<double, std::milli>(graph_setup_stop -
                                                graph_setup_start)
          .count();

  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!check_cuda(cudaGraphLaunch(graph_exec, stream), "graph warmup launch")) {
      cleanup();
      return 1;
    }
  }
  if (!check_cuda(cudaStreamSynchronize(stream), "graph warmup synchronize")) {
    cleanup();
    return 1;
  }

  if (!check_cuda(cudaEventRecord(start, stream), "manual cudaEventRecord(start)")) {
    cleanup();
    return 1;
  }
  const auto manual_wall_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    enqueue_stock_round(device_states, device_drawn, kGames, blocks, kMasterSeed,
                        stream);
  }
  const auto manual_submit_stop = std::chrono::steady_clock::now();
  if (!check_cuda(cudaGetLastError(), "manual round launch") ||
      !check_cuda(cudaEventRecord(stop, stream), "manual cudaEventRecord(stop)") ||
      !check_cuda(cudaEventSynchronize(stop),
                  "manual cudaEventSynchronize(stop)")) {
    cleanup();
    return 1;
  }
  const auto manual_wall_stop = std::chrono::steady_clock::now();

  float manual_gpu_ms_f = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&manual_gpu_ms_f, start, stop),
                  "manual cudaEventElapsedTime")) {
    cleanup();
    return 1;
  }
  const double manual_gpu_ms = manual_gpu_ms_f;
  const double manual_submit_ms =
      std::chrono::duration<double, std::milli>(manual_submit_stop -
                                                manual_wall_start)
          .count();
  const double manual_wall_ms =
      std::chrono::duration<double, std::milli>(manual_wall_stop -
                                                manual_wall_start)
          .count();

  if (!check_cuda(cudaEventRecord(start, stream), "graph cudaEventRecord(start)")) {
    cleanup();
    return 1;
  }
  const auto graph_wall_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    if (!check_cuda(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch")) {
      cleanup();
      return 1;
    }
  }
  const auto graph_submit_stop = std::chrono::steady_clock::now();
  if (!check_cuda(cudaEventRecord(stop, stream), "graph cudaEventRecord(stop)") ||
      !check_cuda(cudaEventSynchronize(stop),
                  "graph cudaEventSynchronize(stop)")) {
    cleanup();
    return 1;
  }
  const auto graph_wall_stop = std::chrono::steady_clock::now();

  float graph_gpu_ms_f = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&graph_gpu_ms_f, start, stop),
                  "graph cudaEventElapsedTime")) {
    cleanup();
    return 1;
  }
  const double graph_gpu_ms = graph_gpu_ms_f;
  const double graph_submit_ms =
      std::chrono::duration<double, std::milli>(graph_submit_stop -
                                                graph_wall_start)
          .count();
  const double graph_wall_ms =
      std::chrono::duration<double, std::milli>(graph_wall_stop -
                                                graph_wall_start)
          .count();

  const double schedule_rounds = static_cast<double>(kGames) * kIterations;
  const double schedule_transitions = schedule_rounds * kSteps;
  const double manual_rounds_per_second =
      schedule_rounds / (manual_gpu_ms / 1000.0);
  const double graph_rounds_per_second =
      schedule_rounds / (graph_gpu_ms / 1000.0);
  const double manual_transitions_per_second =
      schedule_transitions / (manual_gpu_ms / 1000.0);
  const double graph_transitions_per_second =
      schedule_transitions / (graph_gpu_ms / 1000.0);
  const double gpu_speedup = manual_gpu_ms / graph_gpu_ms;
  const double submit_speedup = manual_submit_ms / graph_submit_ms;
  const double wall_speedup = manual_wall_ms / graph_wall_ms;

  std::cout << "device=" << properties.name << " sms=" << properties.multiProcessorCount
            << " threads=" << kThreads << '\n';
  std::cout << "draw_stock_kernel registers/thread=" << attributes.numRegs
            << " local_bytes/thread=" << attributes.localSizeBytes
            << " static_shared_bytes/block=" << attributes.sharedSizeBytes
            << " theoretical_occupancy=" << std::fixed << std::setprecision(2)
            << occupancy * 100.0 << "%\n";
  std::cout << "benchmark_warmups=" << kWarmups
            << " benchmark_iterations=" << kIterations << '\n';
  std::cout << std::fixed << std::setprecision(2)
            << "resident_draw elapsed_ms=" << draw_elapsed_ms
            << " transitions/s=" << draw_transitions_per_second
            << " stock_rounds/s=" << draw_rounds_per_second
            << " avg_draw_kernel_us=" << average_kernel_us << '\n';
  std::cout << "phase_graph nodes=" << graph_nodes
            << " setup_ms=" << graph_setup_ms << '\n';
  std::cout << "manual_round gpu_ms=" << manual_gpu_ms
            << " submit_ms=" << manual_submit_ms
            << " wall_ms=" << manual_wall_ms
            << " stock_rounds/s=" << manual_rounds_per_second
            << " transitions/s=" << manual_transitions_per_second << '\n';
  std::cout << "graph_round gpu_ms=" << graph_gpu_ms
            << " submit_ms=" << graph_submit_ms
            << " wall_ms=" << graph_wall_ms
            << " stock_rounds/s=" << graph_rounds_per_second
            << " transitions/s=" << graph_transitions_per_second << '\n';
  std::cout << "graph_speedup gpu=" << gpu_speedup
            << "x submit=" << submit_speedup
            << "x wall=" << wall_speedup << "x\n";

  cleanup();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--benchmark") {
    return run_benchmark();
  }
  if (argc != 1) {
    std::cerr << "usage: cugo_cuda_state_test [--benchmark]\n";
    return 64;
  }
  return run_differential();
}
