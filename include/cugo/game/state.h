#pragma once

#include <cstddef>
#include <cstdint>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/deal.h"

#if defined(__CUDACC__)
#define CUGO_HOST_DEVICE __host__ __device__
#else
#define CUGO_HOST_DEVICE
#endif

namespace cugo::game {

using core::CardId;
using core::CardMask;

struct ResidentState48 {
  CardMask hand0;
  CardMask hand1;
  CardMask floor;
  CardMask stock;
  std::uint64_t rng_state;
};

struct ResidentStateSoA48 {
  CardMask* hand0;
  CardMask* hand1;
  CardMask* floor;
  CardMask* stock;
  std::uint64_t* rng_state;
};

CUGO_HOST_DEVICE inline ResidentState48 make_resident_state(const InitialDeal48& deal) noexcept {
  return ResidentState48{deal.hand0, deal.hand1, deal.floor, deal.stock, deal.rng_state};
}

CUGO_HOST_DEVICE inline CardId draw_stock_card(CardMask& stock,
                                               std::uint64_t& rng_state) noexcept {
  if (stock == 0) {
    return core::kInvalidCard;
  }

  core::SplitMix64 rng{rng_state};
  const CardId card = draw_uniform_card(stock, rng);
  rng_state = rng.state();
  return card;
}

CUGO_HOST_DEVICE inline CardId draw_stock_card(ResidentState48& state) noexcept {
  return draw_stock_card(state.stock, state.rng_state);
}

CUGO_HOST_DEVICE inline ResidentState48 load_state(const ResidentStateSoA48& states,
                                                   std::size_t game) noexcept {
  return ResidentState48{states.hand0[game], states.hand1[game], states.floor[game],
                         states.stock[game], states.rng_state[game]};
}

CUGO_HOST_DEVICE inline void store_state(const ResidentStateSoA48& states,
                                         std::size_t game,
                                         const ResidentState48& state) noexcept {
  states.hand0[game] = state.hand0;
  states.hand1[game] = state.hand1;
  states.floor[game] = state.floor;
  states.stock[game] = state.stock;
  states.rng_state[game] = state.rng_state;
}

CUGO_HOST_DEVICE inline void store_initial_deal(const ResidentStateSoA48& states,
                                                std::size_t game,
                                                const InitialDeal48& deal) noexcept {
  store_state(states, game, make_resident_state(deal));
}

}  // namespace cugo::game

#undef CUGO_HOST_DEVICE
