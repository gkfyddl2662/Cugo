#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cugo/game/policy50.h"

namespace {

using namespace cugo::game;

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(1);
  }
}

__global__ void prepare_policy_states_kernel(TerminalGameState50* states,
                                              std::uint32_t n) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;

  TerminalGameState50 state = make_terminal_game_state50(
      deal_shin_matgo_50(0x504f4c4943594245ULL + index),
      static_cast<std::uint8_t>(index & 1u));
  const std::uint16_t target = static_cast<std::uint16_t>((index >> 1u) % 24u);
  for (std::uint16_t step = 0; step < target; ++step) {
    if (terminal50_is_finished(state)) break;
    if (!terminal50_has_chongtong_choice(state) &&
        !game50_has_pending_decision(state.special.game) &&
        rollout_hands_exhausted50(state)) break;
    const CanonicalAction50 canonical = canonical_action50(state);
    if (canonical.status != ActionStatus50::kOk) break;
    if (apply_action50(state, canonical.action).status != ActionStatus50::kOk) break;
  }
  states[index] = state;
}

__global__ void encode_policy_kernel(const TerminalGameState50* states,
                                     PolicyPacket50* packets,
                                     std::uint32_t n) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) packets[index] = make_policy_packet50(states[index]);
}

void run_bench(const TerminalGameState50* states,
               PolicyPacket50* packets,
               std::uint32_t games,
               int threads,
               int iterations,
               int warmups,
               const cudaDeviceProp& prop) {
  const int blocks = static_cast<int>((games + static_cast<std::uint32_t>(threads) - 1u) /
                                      static_cast<std::uint32_t>(threads));
  for (int i = 0; i < warmups; ++i)
    encode_policy_kernel<<<blocks, threads>>>(states, packets, games);
  cuda_check(cudaGetLastError(), "encode_policy_kernel warmup launch");
  cuda_check(cudaDeviceSynchronize(), "encode_policy_kernel warmup sync");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
  cuda_check(cudaEventRecord(start), "cudaEventRecord start");
  for (int i = 0; i < iterations; ++i)
    encode_policy_kernel<<<blocks, threads>>>(states, packets, games);
  cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
  cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
  cuda_check(cudaGetLastError(), "encode_policy_kernel launch");

  float elapsed_ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
  cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
  cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");

  cudaFuncAttributes attr{};
  cuda_check(cudaFuncGetAttributes(&attr, encode_policy_kernel), "cudaFuncGetAttributes");
  int active_blocks = 0;
  cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &active_blocks, encode_policy_kernel, threads, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

  const double packets_total = static_cast<double>(games) * iterations;
  const double packets_per_s = packets_total / (static_cast<double>(elapsed_ms) * 1.0e-3);
  const double ns_per_packet = 1.0e9 / packets_per_s;
  const double occupancy = 100.0 * static_cast<double>(active_blocks * threads) /
                           static_cast<double>(prop.maxThreadsPerMultiProcessor);

  std::cout << "threads=" << threads
            << " blocks=" << blocks
            << " elapsed_ms=" << std::fixed << std::setprecision(3) << elapsed_ms
            << " packets_per_s=" << std::setprecision(3) << packets_per_s
            << " ns_per_packet=" << std::setprecision(4) << ns_per_packet
            << " regs_per_thread=" << attr.numRegs
            << " local_bytes_per_thread=" << attr.localSizeBytes
            << " active_blocks_per_sm=" << active_blocks
            << " occupancy=" << std::setprecision(2) << occupancy << "%\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t games = 1u << 20;
  int iterations = 256;
  int warmups = 8;
  if (argc > 1) games = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
  if (argc > 2) iterations = std::atoi(argv[2]);
  if (argc > 3) warmups = std::atoi(argv[3]);
  if (games == 0 || iterations <= 0 || warmups < 0) return 2;

  int device_id = 0;
  cudaDeviceProp prop{};
  cuda_check(cudaGetDevice(&device_id), "cudaGetDevice");
  cuda_check(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");

  TerminalGameState50* states = nullptr;
  PolicyPacket50* packets = nullptr;
  cuda_check(cudaMalloc(&states, sizeof(TerminalGameState50) * games), "cudaMalloc states");
  cuda_check(cudaMalloc(&packets, sizeof(PolicyPacket50) * games), "cudaMalloc packets");

  constexpr int kPrepareThreads = 256;
  const int prepare_blocks = static_cast<int>((games + kPrepareThreads - 1u) / kPrepareThreads);
  prepare_policy_states_kernel<<<prepare_blocks, kPrepareThreads>>>(states, games);
  cuda_check(cudaGetLastError(), "prepare_policy_states_kernel launch");
  cuda_check(cudaDeviceSynchronize(), "prepare_policy_states_kernel sync");

  std::cout << "device=" << prop.name
            << " sms=" << prop.multiProcessorCount
            << " packets=" << games
            << " iterations=" << iterations
            << " warmups=" << warmups
            << " packet_bytes=" << sizeof(PolicyPacket50)
            << " state_bytes=" << sizeof(TerminalGameState50) << '\n';

  for (int threads : {128, 256, 512})
    run_bench(states, packets, games, threads, iterations, warmups, prop);

  const int final_blocks = static_cast<int>((games + 255u) / 256u);
  encode_policy_kernel<<<final_blocks, 256>>>(states, packets, games);
  cuda_check(cudaGetLastError(), "encode_policy_kernel final launch");
  cuda_check(cudaDeviceSynchronize(), "encode_policy_kernel final sync");

  std::vector<PolicyPacket50> host(games);
  cuda_check(cudaMemcpy(host.data(), packets, sizeof(PolicyPacket50) * games,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy packets");

  std::uint64_t ready = 0;
  std::uint64_t legal_total = 0;
  std::uint64_t go_stop = 0;
  std::uint64_t chongtong = 0;
  for (const PolicyPacket50& packet : host) {
    const auto decision = static_cast<PolicyDecision50>(packet.observation.decision_kind);
    if (decision != PolicyDecision50::kNone) ++ready;
    if (decision == PolicyDecision50::kGoStop) ++go_stop;
    if (decision == PolicyDecision50::kChongtong) ++chongtong;
    legal_total += policy_action_count50(packet.legal_actions);
  }

  std::cout << "snapshot ready=" << ready
            << " go_stop=" << go_stop
            << " chongtong=" << chongtong
            << " avg_legal=" << std::fixed << std::setprecision(3)
            << static_cast<double>(legal_total) / static_cast<double>(games) << '\n';

  cuda_check(cudaFree(packets), "cudaFree packets");
  cuda_check(cudaFree(states), "cudaFree states");
  return 0;
}
