#pragma once

#include <cstdint>

#include "cugo/game/settlement50.h"

#if defined(__CUDACC__)
#define CUGO_HD __host__ __device__
#else
#define CUGO_HD
#endif

namespace cugo::game {

using core::CardId;
using core::CardMask;

#include "cugo/game/detail/rollout50_types.inc"
#include "cugo/game/detail/rollout50_apply.inc"
#include "cugo/game/detail/rollout50_canonicalize.inc"
#include "cugo/game/detail/rollout50_policy.inc"
#include "cugo/game/detail/rollout50_driver.inc"

}  // namespace cugo::game

#undef CUGO_HD
