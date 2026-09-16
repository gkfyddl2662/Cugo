#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "cugo/game/torch50.h"

namespace {

using cugo::game::TorchEnv50;
using cugo::game::TorchPacket50;
using cugo::game::TorchStepResult50;

static_assert(std::is_trivially_copyable_v<TorchEnv50>);

constexpr int kThreads = 256;

void check_cuda_1d(const torch::Tensor& tensor,
                   at::ScalarType dtype,
                   const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

void check_states(const torch::Tensor& states) {
  TORCH_CHECK(states.is_cuda(), "states must be a CUDA tensor");
  TORCH_CHECK(states.is_contiguous(), "states must be contiguous");
  TORCH_CHECK(states.scalar_type() == at::kByte,
              "states must have dtype torch.uint8");
  TORCH_CHECK(states.dim() == 2, "states must have shape [N, STATE_BYTES]");
  TORCH_CHECK(states.size(1) == static_cast<std::int64_t>(sizeof(TorchEnv50)),
              "states second dimension must equal STATE_BYTES");
}

__global__ void create_kernel(TorchEnv50* envs,
                              const std::int64_t* seeds,
                              const std::int64_t* first_players,
                              std::int64_t count) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  envs[i] = cugo::game::make_torch_env50(
      static_cast<std::uint64_t>(seeds[i]),
      static_cast<std::uint8_t>(first_players[i] & 1ll));
}

__global__ void observe_kernel(const TorchEnv50* envs,
                               float* features,
                               bool* legal,
                               std::uint8_t* players,
                               bool* done,
                               std::int16_t* status,
                               std::int64_t count) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;

  const TorchPacket50 packet = cugo::game::make_torch_packet50(envs[i]);
  cugo::game::encode_torch_packet50(
      packet, features + i * cugo::game::kTorch50FeatureCount);

  bool* legal_row = legal + i * cugo::game::kTorch50ActionCount;
  for (std::uint16_t action = 0; action < cugo::game::kTorch50ActionCount;
       ++action) {
    legal_row[action] =
        cugo::game::has_torch_action50(packet.legal_actions, action);
  }
  players[i] = packet.decision_player;
  done[i] = envs[i].done != 0;
  status[i] = static_cast<std::int16_t>(packet.status);
}

__global__ void step_kernel(TorchEnv50* envs,
                            const std::int64_t* actions,
                            float* rewards,
                            bool* done,
                            bool* nagari,
                            std::int16_t* status,
                            bool* primary_committed,
                            std::int64_t count) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;

  const std::int64_t raw = actions[i];
  const std::uint16_t action =
      raw >= 0 && raw < cugo::game::kTorch50ActionCount
          ? static_cast<std::uint16_t>(raw)
          : cugo::game::kInvalidTorch50Action;
  const TorchStepResult50 result = cugo::game::torch_step50(envs[i], action);
  rewards[i] = static_cast<float>(result.reward0);
  done[i] = result.done != 0;
  nagari[i] = result.nagari != 0;
  status[i] = static_cast<std::int16_t>(result.status);
  primary_committed[i] = result.primary_committed != 0;
}

}  // namespace

torch::Tensor cugo_torch50_create_cuda(torch::Tensor seeds,
                                        torch::Tensor first_players) {
  check_cuda_1d(seeds, at::kLong, "seeds");
  check_cuda_1d(first_players, at::kLong, "first_players");
  TORCH_CHECK(seeds.numel() == first_players.numel(),
              "seeds and first_players must have the same length");
  TORCH_CHECK(seeds.device() == first_players.device(),
              "seeds and first_players must be on the same CUDA device");

  const c10::cuda::CUDAGuard guard(seeds.device());
  const std::int64_t count = seeds.numel();
  auto states = torch::empty(
      {count, static_cast<std::int64_t>(sizeof(TorchEnv50))},
      torch::TensorOptions().device(seeds.device()).dtype(at::kByte));

  if (count != 0) {
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    create_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<TorchEnv50*>(states.data_ptr<std::uint8_t>()),
        seeds.data_ptr<std::int64_t>(),
        first_players.data_ptr<std::int64_t>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return states;
}

std::vector<torch::Tensor> cugo_torch50_observe_cuda(torch::Tensor states) {
  check_states(states);
  const c10::cuda::CUDAGuard guard(states.device());
  const std::int64_t count = states.size(0);

  auto features = torch::empty(
      {count, static_cast<std::int64_t>(cugo::game::kTorch50FeatureCount)},
      torch::TensorOptions().device(states.device()).dtype(at::kFloat));
  auto legal = torch::empty(
      {count, static_cast<std::int64_t>(cugo::game::kTorch50ActionCount)},
      torch::TensorOptions().device(states.device()).dtype(at::kBool));
  auto players = torch::empty(
      {count}, torch::TensorOptions().device(states.device()).dtype(at::kByte));
  auto done = torch::empty(
      {count}, torch::TensorOptions().device(states.device()).dtype(at::kBool));
  auto status = torch::empty(
      {count}, torch::TensorOptions().device(states.device()).dtype(at::kShort));

  if (count != 0) {
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    observe_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const TorchEnv50*>(states.data_ptr<std::uint8_t>()),
        features.data_ptr<float>(), legal.data_ptr<bool>(),
        players.data_ptr<std::uint8_t>(), done.data_ptr<bool>(),
        status.data_ptr<std::int16_t>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {features, legal, players, done, status};
}

std::vector<torch::Tensor> cugo_torch50_step_cuda(torch::Tensor states,
                                                   torch::Tensor actions) {
  check_states(states);
  check_cuda_1d(actions, at::kLong, "actions");
  TORCH_CHECK(states.size(0) == actions.numel(),
              "states and actions must have the same batch size");
  TORCH_CHECK(states.device() == actions.device(),
              "states and actions must be on the same CUDA device");

  const c10::cuda::CUDAGuard guard(states.device());
  const std::int64_t count = states.size(0);
  auto rewards = torch::empty(
      {count}, torch::TensorOptions().device(states.device()).dtype(at::kFloat));
  auto done = torch::empty(
      {count}, torch::TensorOptions().device(states.device()).dtype(at::kBool));
  auto nagari = torch::empty(
      {count}, torch::TensorOptions().device(states.device()).dtype(at::kBool));
  auto status = torch::empty(
      {count}, torch::TensorOptions().device(states.device()).dtype(at::kShort));
  auto committed = torch::empty(
      {count}, torch::TensorOptions().device(states.device()).dtype(at::kBool));

  if (count != 0) {
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    step_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<TorchEnv50*>(states.data_ptr<std::uint8_t>()),
        actions.data_ptr<std::int64_t>(), rewards.data_ptr<float>(),
        done.data_ptr<bool>(), nagari.data_ptr<bool>(),
        status.data_ptr<std::int16_t>(), committed.data_ptr<bool>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {rewards, done, nagari, status, committed};
}
