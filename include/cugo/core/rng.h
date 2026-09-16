#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::core {

class SplitMix64 {
 public:
  CUGO_HOST_DEVICE explicit constexpr SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

  CUGO_HOST_DEVICE std::uint64_t next_u64() noexcept {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  CUGO_HOST_DEVICE std::uint32_t next_u32() noexcept {
    return static_cast<std::uint32_t>(next_u64() >> 32);
  }

  CUGO_HOST_DEVICE std::uint64_t state() const noexcept { return state_; }

 private:
  std::uint64_t state_;
};

CUGO_HOST_DEVICE inline std::uint32_t uniform_bounded(SplitMix64& rng,
                                                       std::uint32_t bound) noexcept {
  if (bound <= 1u) {
    return 0u;
  }

  for (;;) {
    const std::uint32_t x = rng.next_u32();
    const std::uint64_t product = static_cast<std::uint64_t>(x) * bound;
    const std::uint32_t low = static_cast<std::uint32_t>(product);

    if (low < bound) {
      const std::uint32_t threshold = (0u - bound) % bound;
      if (low < threshold) {
        continue;
      }
    }

    return static_cast<std::uint32_t>(product >> 32);
  }
}

CUGO_HOST_DEVICE inline std::uint64_t derive_seed(std::uint64_t master_seed,
                                                   std::uint64_t stream_id) noexcept {
  SplitMix64 seeder{master_seed + stream_id * 0x9e3779b97f4a7c15ULL};
  return seeder.next_u64();
}

}  // namespace cugo::core

#undef CUGO_HOST_DEVICE
