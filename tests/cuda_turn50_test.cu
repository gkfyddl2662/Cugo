#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cugo/core/card.h"
#include "cugo/core/rng.h"
#include "cugo/game/deal.h"
#include "cugo/game/turn50.h"

namespace {

using cugo::game::Resolve50Result;
using cugo::game::Resolve50Status;
using cugo::game::TurnState50;
using cugo::game::TurnStateSoA50;

constexpr int kThreads = 256;
constexpr int kTargetedGames = 2;
constexpr int kU64Fields = 8;
constexpr int kU16Fields = 5;
constexpr int kU8Fields = 4;

struct BonusSnapshot {
  std::uint8_t attempted;
  std::uint8_t replacement;
  std::uint8_t pi_steal_count;
};

struct ResolveSnapshot {
  std::uint64_t captured_cards;
  std::uint8_t status;
  std::uint8_t events;
  std::uint8_t captured_own_ppuk;
  std::uint8_t captured_opponent_ppuk;
};

TurnStateSoA50 make_view(std::uint64_t* u64,
                         std::uint16_t* u16,
                         std::uint8_t* u8,
                         std::size_t games) {
  return TurnStateSoA50{
      u64 + games * 0,
      u64 + games * 1,
      u64 + games * 2,
      u64 + games * 3,
      u64 + games * 4,
      u64 + games * 5,
      u64 + games * 6,
      u64 + games * 7,
      u16 + games * 0,
      u16 + games * 1,
      u16 + games * 2,
      u16 + games * 3,
      u16 + games * 4,
      u8 + games * 0,
      u8 + games * 1,
      u8 + games * 2,
      u8 + games * 3,
  };
}

__host__ __device__ TurnState50 make_targeted_state(int game) {
  using namespace cugo::core;
  using namespace cugo::game;

  TurnState50 state{};
  state.rng_state = 0x123456789abcdef0ULL;
  state.phase = Turn50Phase::kResolve;

  if (game == 0) {
    const CardMask used = card_bit(0) | card_bit(1) | card_bit(2) |
                          kBonusCardMask;
    state.floor = card_bit(0);
    state.stock = kShinMatgoDeckMask & ~used;
    state.pending_bonus_mask = kBonusCardMask;
    state.pending_played = 1;
    state.pending_drawn = 2;
    state.actor = 0;
  } else {
    const CardMask floor = card_bit(0) | card_bit(1) | card_bit(2) |
                           kBonusCardMask;
    const CardMask used = floor | card_bit(3) | card_bit(4);
    state.floor = floor;
    state.stock = kShinMatgoDeckMask & ~used;
    state.ppuk_months = 1u;
    state.bonus2_ppuk_months = 1u;
    state.bonus3_ppuk_months = 1u;
    state.pending_played = 3;
    state.pending_drawn = 4;
    state.actor = 1;
  }
  return state;
}

__global__ void init_turn50_kernel(TurnStateSoA50 states,
                                   std::uint8_t* errors,
                                   int games,
                                   std::uint64_t master_seed) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) return;

  TurnState50 state{};
  if (game < kTargetedGames) {
    state = make_targeted_state(game);
  } else {
    const std::uint64_t seed = cugo::core::derive_seed(
        master_seed, static_cast<std::uint64_t>(game));
    state = cugo::game::make_turn_state50(
        cugo::game::deal_shin_matgo_50(seed),
        static_cast<std::uint8_t>(game & 1));
  }

  if (!cugo::game::is_valid_turn_state50(state)) errors[game] = 1;
  cugo::game::store_turn_state50(states, static_cast<std::size_t>(game), state);
}

__global__ void bonus_play50_kernel(TurnStateSoA50 states,
                                    BonusSnapshot* snapshots,
                                    std::uint8_t* errors,
                                    int games) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) return;
  if (game < kTargetedGames) {
    snapshots[game] = BonusSnapshot{0, cugo::core::kInvalidCard, 0};
    return;
  }

  auto state = cugo::game::load_turn_state50(states, static_cast<std::size_t>(game));
  const cugo::core::CardMask bonuses =
      cugo::game::active_hand50(state) & cugo::core::kBonusCardMask;
  if (bonuses == 0) {
    snapshots[game] = BonusSnapshot{0, cugo::core::kInvalidCard, 0};
    return;
  }

  const auto result = cugo::game::play_bonus_for_turn50(
      state, cugo::core::first_card(bonuses));
  if (result.status != cugo::game::Turn50Status::kOk ||
      !cugo::game::is_valid_turn_state50(state)) {
    errors[game] = 2;
  }
  snapshots[game] = BonusSnapshot{1, result.replacement, result.pi_steal_count};
  cugo::game::store_turn_state50(states, static_cast<std::size_t>(game), state);
}

__global__ void regular_play50_kernel(TurnStateSoA50 states,
                                      std::uint8_t* errors,
                                      int games) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games || game < kTargetedGames) return;

  auto state = cugo::game::load_turn_state50(states, static_cast<std::size_t>(game));
  const cugo::core::CardMask standards =
      cugo::game::active_hand50(state) & cugo::core::kStandardDeckMask;
  const cugo::core::CardId card = cugo::core::first_card(standards);
  if (card == cugo::core::kInvalidCard ||
      cugo::game::begin_regular_play50(state, card) !=
          cugo::game::Turn50Status::kOk ||
      !cugo::game::is_valid_turn_state50(state)) {
    errors[game] = 3;
  }
  cugo::game::store_turn_state50(states, static_cast<std::size_t>(game), state);
}

__global__ void draw50_kernel(TurnStateSoA50 states,
                              std::uint8_t* errors,
                              int games) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games || game < kTargetedGames) return;

  auto state = cugo::game::load_turn_state50(states, static_cast<std::size_t>(game));
  if (cugo::game::draw_for_turn50(state) != cugo::game::Turn50Status::kOk ||
      !cugo::game::is_valid_turn_state50(state)) {
    errors[game] = 4;
  }
  cugo::game::store_turn_state50(states, static_cast<std::size_t>(game), state);
}

__global__ void resolve50_kernel(TurnStateSoA50 states,
                                 ResolveSnapshot* results,
                                 std::uint8_t* errors,
                                 int games) {
  const int game = blockIdx.x * blockDim.x + threadIdx.x;
  if (game >= games) return;

  auto state = cugo::game::load_turn_state50(states, static_cast<std::size_t>(game));
  const Resolve50Result result = cugo::game::resolve_turn50(state);
  if (result.status == Resolve50Status::kOk) {
    if (!cugo::game::is_valid_turn_state50(state) ||
        state.phase != cugo::game::Turn50Phase::kPlay) {
      errors[game] = 5;
    }
  } else if (result.status == Resolve50Status::kChoiceRequired) {
    if (!cugo::game::is_valid_turn_state50(state) ||
        state.phase != cugo::game::Turn50Phase::kResolve) {
      errors[game] = 6;
    }
  } else {
    errors[game] = 7;
  }

  results[game] = ResolveSnapshot{
      result.captured_cards,
      static_cast<std::uint8_t>(result.status),
      result.events,
      result.captured_own_ppuk,
      result.captured_opponent_ppuk,
  };
  cugo::game::store_turn_state50(states, static_cast<std::size_t>(game), state);
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return true;
  std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool same_state(const TurnState50& a, const TurnState50& b) {
  return a.hand0 == b.hand0 && a.hand1 == b.hand1 && a.floor == b.floor &&
         a.stock == b.stock && a.captured0 == b.captured0 &&
         a.captured1 == b.captured1 && a.rng_state == b.rng_state &&
         a.pending_bonus_mask == b.pending_bonus_mask &&
         a.ppuk_months == b.ppuk_months &&
         a.ppuk_owner1_months == b.ppuk_owner1_months &&
         a.bonus2_ppuk_months == b.bonus2_ppuk_months &&
         a.bonus3_ppuk_months == b.bonus3_ppuk_months &&
         a.turn_index == b.turn_index && a.actor == b.actor &&
         a.phase == b.phase && a.pending_played == b.pending_played &&
         a.pending_drawn == b.pending_drawn;
}

bool same_result(const ResolveSnapshot& gpu, const Resolve50Result& cpu) {
  return gpu.captured_cards == cpu.captured_cards &&
         gpu.status == static_cast<std::uint8_t>(cpu.status) &&
         gpu.events == cpu.events &&
         gpu.captured_own_ppuk == cpu.captured_own_ppuk &&
         gpu.captured_opponent_ppuk == cpu.captured_opponent_ppuk;
}

}  // namespace

int main() {
  constexpr int kGames = 1 << 16;
  constexpr std::uint64_t kMasterSeed = 0x50305f6770755f31ULL;
  const int blocks = (kGames + kThreads - 1) / kThreads;
  const std::size_t games = static_cast<std::size_t>(kGames);

  std::uint64_t* device_u64 = nullptr;
  std::uint16_t* device_u16 = nullptr;
  std::uint8_t* device_u8 = nullptr;
  std::uint8_t* device_errors = nullptr;
  BonusSnapshot* device_bonus = nullptr;
  ResolveSnapshot* device_results = nullptr;

  auto cleanup = [&]() {
    cudaFree(device_u64);
    cudaFree(device_u16);
    cudaFree(device_u8);
    cudaFree(device_errors);
    cudaFree(device_bonus);
    cudaFree(device_results);
  };

  if (!check_cuda(cudaMalloc(&device_u64, games * kU64Fields * sizeof(std::uint64_t)), "cudaMalloc(u64)") ||
      !check_cuda(cudaMalloc(&device_u16, games * kU16Fields * sizeof(std::uint16_t)), "cudaMalloc(u16)") ||
      !check_cuda(cudaMalloc(&device_u8, games * kU8Fields * sizeof(std::uint8_t)), "cudaMalloc(u8)") ||
      !check_cuda(cudaMalloc(&device_errors, games * sizeof(std::uint8_t)), "cudaMalloc(errors)") ||
      !check_cuda(cudaMalloc(&device_bonus, games * sizeof(BonusSnapshot)), "cudaMalloc(bonus)") ||
      !check_cuda(cudaMalloc(&device_results, games * sizeof(ResolveSnapshot)), "cudaMalloc(results)") ||
      !check_cuda(cudaMemset(device_errors, 0, games * sizeof(std::uint8_t)), "cudaMemset(errors)")) {
    cleanup();
    return 1;
  }

  const TurnStateSoA50 device_states = make_view(device_u64, device_u16, device_u8, games);
  init_turn50_kernel<<<blocks, kThreads>>>(device_states, device_errors, kGames, kMasterSeed);
  bonus_play50_kernel<<<blocks, kThreads>>>(device_states, device_bonus, device_errors, kGames);
  regular_play50_kernel<<<blocks, kThreads>>>(device_states, device_errors, kGames);
  draw50_kernel<<<blocks, kThreads>>>(device_states, device_errors, kGames);
  resolve50_kernel<<<blocks, kThreads>>>(device_states, device_results, device_errors, kGames);

  if (!check_cuda(cudaGetLastError(), "turn50 launch") ||
      !check_cuda(cudaDeviceSynchronize(), "turn50 synchronize")) {
    cleanup();
    return 1;
  }

  std::vector<std::uint64_t> host_u64(games * kU64Fields);
  std::vector<std::uint16_t> host_u16(games * kU16Fields);
  std::vector<std::uint8_t> host_u8(games * kU8Fields);
  std::vector<std::uint8_t> host_errors(games);
  std::vector<BonusSnapshot> host_bonus(games);
  std::vector<ResolveSnapshot> host_results(games);

  if (!check_cuda(cudaMemcpy(host_u64.data(), device_u64,
                             host_u64.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy(u64)") ||
      !check_cuda(cudaMemcpy(host_u16.data(), device_u16,
                             host_u16.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy(u16)") ||
      !check_cuda(cudaMemcpy(host_u8.data(), device_u8,
                             host_u8.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost), "cudaMemcpy(u8)") ||
      !check_cuda(cudaMemcpy(host_errors.data(), device_errors,
                             host_errors.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost), "cudaMemcpy(errors)") ||
      !check_cuda(cudaMemcpy(host_bonus.data(), device_bonus,
                             host_bonus.size() * sizeof(BonusSnapshot), cudaMemcpyDeviceToHost), "cudaMemcpy(bonus)") ||
      !check_cuda(cudaMemcpy(host_results.data(), device_results,
                             host_results.size() * sizeof(ResolveSnapshot), cudaMemcpyDeviceToHost), "cudaMemcpy(results)")) {
    cleanup();
    return 1;
  }
  cleanup();

  const TurnStateSoA50 gpu_view = make_view(host_u64.data(), host_u16.data(), host_u8.data(), games);
  int resolved = 0;
  int choice_required = 0;
  int initial_bonus = 0;
  int hand_bonus = 0;
  int stock_bonus = 0;
  int ppuk_with_bonus = 0;

  for (int game = 0; game < kGames; ++game) {
    if (host_errors[game] != 0) {
      std::cerr << "GPU turn50 error at game " << game
                << " code=" << static_cast<unsigned>(host_errors[game]) << '\n';
      return 2;
    }

    TurnState50 cpu{};
    BonusSnapshot cpu_bonus{0, cugo::core::kInvalidCard, 0};
    if (game < kTargetedGames) {
      cpu = make_targeted_state(game);
    } else {
      const std::uint64_t seed = cugo::core::derive_seed(
          kMasterSeed, static_cast<std::uint64_t>(game));
      const auto deal = cugo::game::deal_shin_matgo_50(seed);
      if ((deal.floor & cugo::core::kBonusCardMask) != 0) ++initial_bonus;
      cpu = cugo::game::make_turn_state50(deal, static_cast<std::uint8_t>(game & 1));

      const cugo::core::CardMask bonuses =
          cugo::game::active_hand50(cpu) & cugo::core::kBonusCardMask;
      if (bonuses != 0) {
        ++hand_bonus;
        const auto br = cugo::game::play_bonus_for_turn50(
            cpu, cugo::core::first_card(bonuses));
        if (br.status != cugo::game::Turn50Status::kOk) return 3;
        cpu_bonus = BonusSnapshot{1, br.replacement, br.pi_steal_count};
      }

      const cugo::core::CardMask standards =
          cugo::game::active_hand50(cpu) & cugo::core::kStandardDeckMask;
      if (cugo::game::begin_regular_play50(cpu, cugo::core::first_card(standards)) !=
              cugo::game::Turn50Status::kOk ||
          cugo::game::draw_for_turn50(cpu) != cugo::game::Turn50Status::kOk) {
        return 4;
      }
      if (cpu.pending_bonus_mask != 0) ++stock_bonus;
    }

    if (host_bonus[game].attempted != cpu_bonus.attempted ||
        host_bonus[game].replacement != cpu_bonus.replacement ||
        host_bonus[game].pi_steal_count != cpu_bonus.pi_steal_count) {
      std::cerr << "CPU/GPU bonus action mismatch at game " << game << '\n';
      return 5;
    }

    const Resolve50Result cpu_result = cugo::game::resolve_turn50(cpu);
    if (cpu_result.status == Resolve50Status::kOk) {
      ++resolved;
      if ((cpu_result.events & cugo::game::kResolve50EventPpuk) != 0 &&
          (cpu.floor & cugo::core::kBonusCardMask) != 0) {
        ++ppuk_with_bonus;
      }
    } else if (cpu_result.status == Resolve50Status::kChoiceRequired) {
      ++choice_required;
    } else {
      std::cerr << "Unexpected CPU resolve50 status at game " << game << '\n';
      return 6;
    }

    const TurnState50 gpu = cugo::game::load_turn_state50(
        gpu_view, static_cast<std::size_t>(game));
    if (!same_state(gpu, cpu) || !same_result(host_results[game], cpu_result) ||
        !cugo::game::is_valid_turn_state50(gpu)) {
      std::cerr << "CPU/GPU turn50 mismatch at game " << game << '\n';
      return 7;
    }
  }

  if (resolved == 0 || choice_required == 0 || initial_bonus == 0 ||
      hand_bonus == 0 || stock_bonus == 0 || ppuk_with_bonus == 0) {
    std::cerr << "Coverage missing resolved=" << resolved
              << " choice_required=" << choice_required
              << " initial_bonus=" << initial_bonus
              << " hand_bonus=" << hand_bonus
              << " stock_bonus=" << stock_bonus
              << " ppuk_with_bonus=" << ppuk_with_bonus << '\n';
    return 8;
  }

  std::cout << "cugo_cuda_turn50_test: PASS (" << kGames
            << " games; resolved=" << resolved
            << " choice_required=" << choice_required
            << " initial_bonus=" << initial_bonus
            << " hand_bonus=" << hand_bonus
            << " stock_bonus=" << stock_bonus
            << " ppuk_with_bonus=" << ppuk_with_bonus
            << "; targeted_ppuk_bonus_roundtrip=2)\n";
  return 0;
}
