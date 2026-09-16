#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/hwatu.h"

namespace {

struct ScoreSample {
  std::uint64_t cards;
  cugo::game::ScoreBreakdown score;
};

__global__ void evaluate_scores(ScoreSample* samples,
                                int n,
                                std::uint64_t seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  cugo::core::SplitMix64 rng{seed + static_cast<std::uint64_t>(i)};
  const std::uint64_t cards = rng.next_u64() & cugo::core::kFullDeckMask;
  const cugo::game::ScoreOptions options{(i & 1) != 0};
  samples[i] = ScoreSample{cards, cugo::game::score_captured(cards, options)};
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool same_score(const cugo::game::ScoreBreakdown& a,
                const cugo::game::ScoreBreakdown& b) {
  return a.bright_count == b.bright_count &&
         a.animal_count == b.animal_count &&
         a.ribbon_count == b.ribbon_count && a.pi_units == b.pi_units &&
         a.bright_points == b.bright_points &&
         a.animal_points == b.animal_points &&
         a.ribbon_points == b.ribbon_points && a.pi_points == b.pi_points &&
         a.flags == b.flags && a.total_points == b.total_points;
}

void print_score(const char* name, const cugo::game::ScoreBreakdown& score) {
  std::cerr << name << " bright=" << static_cast<unsigned>(score.bright_count)
            << " animal=" << static_cast<unsigned>(score.animal_count)
            << " ribbon=" << static_cast<unsigned>(score.ribbon_count)
            << " pi_units=" << static_cast<unsigned>(score.pi_units)
            << " points=" << static_cast<unsigned>(score.bright_points) << '/'
            << static_cast<unsigned>(score.animal_points) << '/'
            << static_cast<unsigned>(score.ribbon_points) << '/'
            << static_cast<unsigned>(score.pi_points)
            << " flags=" << static_cast<unsigned>(score.flags)
            << " total=" << static_cast<unsigned>(score.total_points) << '\n';
}

}  // namespace

int main() {
  constexpr int kSamples = 1 << 16;
  constexpr int kThreads = 256;
  constexpr std::uint64_t kSeed = 0x73636f72655f6770ULL;

  ScoreSample* device = nullptr;
  if (!check_cuda(cudaMalloc(&device, sizeof(ScoreSample) * kSamples),
                  "cudaMalloc")) {
    return 1;
  }

  evaluate_scores<<<(kSamples + kThreads - 1) / kThreads, kThreads>>>(
      device, kSamples, kSeed);
  if (!check_cuda(cudaGetLastError(), "score kernel launch") ||
      !check_cuda(cudaDeviceSynchronize(), "score kernel synchronize")) {
    cudaFree(device);
    return 1;
  }

  std::vector<ScoreSample> gpu(kSamples);
  if (!check_cuda(cudaMemcpy(gpu.data(),
                             device,
                             sizeof(ScoreSample) * kSamples,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy")) {
    cudaFree(device);
    return 1;
  }
  cudaFree(device);

  for (int i = 0; i < kSamples; ++i) {
    cugo::core::SplitMix64 rng{kSeed + static_cast<std::uint64_t>(i)};
    const std::uint64_t cards = rng.next_u64() & cugo::core::kFullDeckMask;
    const cugo::game::ScoreOptions options{(i & 1) != 0};
    const auto cpu = cugo::game::score_captured(cards, options);

    if (gpu[i].cards != cards || !same_score(gpu[i].score, cpu)) {
      std::cerr << "CPU/GPU score mismatch at sample " << i
                << " cards=0x" << std::hex << cards << std::dec
                << " gukjin_as_pi=" << options.gukjin_as_double_pi << '\n';
      print_score("gpu", gpu[i].score);
      print_score("cpu", cpu);
      return 2;
    }
  }

  std::cout << "cugo_cuda_score_test: PASS (" << kSamples
            << " scoring differential samples)\n";
  return 0;
}
