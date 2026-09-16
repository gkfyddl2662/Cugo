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

}  // namespace cugo::core

#undef CUGO_HOST_DEVICE
