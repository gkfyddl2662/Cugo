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
constexpr std::int16_t kInvalidIndexedStatus = -1;

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

void check_same_device(const torch::Tensor& first,
                       const torch::Tensor& second,
                       const char* first_name,
                       const char* second_name) {
  TORCH_CHECK(first.device() == second.device(), first_name, " and ",
              second_name, " must be on the same CUDA device");
}

__device__ void write_observation(const TorchEnv50& env,
                                  float* features,
                                  bool* legal,
                                  std::uint8_t* player,
                                  bool* done,
                                  std::int16_t* status) {
  const TorchPacket50 packet = cugo::game::make_torch_packet50(env);
  cugo::game::encode_torch_packet50(packet, features);

  for (std::uint16_t action = 0; action < cugo::game::kTorch50ActionCount;
       ++action) {
    legal[action] = cugo::game::has_torch_action50(packet.legal_actions, action);
  }
  *player = packet.decision_player;
  *done = env.done != 0;
  *status = static_cast<std::int16_t>(packet.status);
}

__device__ void write_invalid_observation(float* features,
                                          bool* legal,
                                          std::uint8_t* player,
                                          bool* done,
                                          std::int16_t* status) {
  for (std::uint16_t feature = 0; feature < cugo::game::kTorch50FeatureCount;
       ++feature) {
    features[feature] = 0.0f;
  }
  for (std::uint16_t action = 0; action < cugo::game::kTorch50ActionCount;
       ++action) {
    legal[action] = false;
  }
  *player = 0;
  *done = true;
  *status = kInvalidIndexedStatus;
}

__device__ void write_step_result(TorchEnv50& env,
                                  std::int64_t raw_action,
                                  float* reward,
                                  bool* done,
                                  bool* nagari,
                                  std::int16_t* status,
                                  bool* primary_committed) {
  const std::uint16_t action =
      raw_action >= 0 && raw_action < cugo::game::kTorch50ActionCount
          ? static_cast<std::uint16_t>(raw_action)
          : cugo::game::kInvalidTorch50Action;
  const TorchStepResult50 result = cugo::game::torch_step50(env, action);
  *reward = static_cast<float>(result.reward0);
  *done = result.done != 0;
  *nagari = result.nagari != 0;
  *status = static_cast<std::int16_t>(result.status);
  *primary_committed = result.primary_committed != 0;
}

__device__ void write_invalid_step(float* reward,
                                   bool* done,
                                   bool* nagari,
                                   std::int16_t* status,
                                   bool* primary_committed) {
  *reward = 0.0f;
  *done = true;
  *nagari = false;
  *status = kInvalidIndexedStatus;
  *primary_committed = false;
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

  write_observation(
      envs[i], features + i * cugo::game::kTorch50FeatureCount,
      legal + i * cugo::game::kTorch50ActionCount, players + i, done + i,
      status + i);
}

__global__ void observe_indexed_kernel(const TorchEnv50* envs,
                                       const std::int64_t* env_ids,
                                       float* features,
                                       bool* legal,
                                       std::uint8_t* players,
                                       bool* done,
                                       std::int16_t* status,
                                       std::int64_t count,
                                       std::int64_t env_count) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;

  const std::int64_t env_id = env_ids[i];
  float* feature_row = features + i * cugo::game::kTorch50FeatureCount;
  bool* legal_row = legal + i * cugo::game::kTorch50ActionCount;
  if (env_id < 0 || env_id >= env_count) {
    write_invalid_observation(feature_row, legal_row, players + i, done + i,
                              status + i);
    return;
  }

  write_observation(envs[env_id], feature_row, legal_row, players + i, done + i,
                    status + i);
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

  write_step_result(envs[i], actions[i], rewards + i, done + i, nagari + i,
                    status + i, primary_committed + i);
}

__global__ void step_indexed_kernel(TorchEnv50* envs,
                                    const std::int64_t* env_ids,
                                    const std::int64_t* actions,
                                    float* rewards,
                                    bool* done,
                                    bool* nagari,
                                    std::int16_t* status,
                                    bool* primary_committed,
                                    std::int64_t count,
                                    std::int64_t env_count) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;

  const std::int64_t env_id = env_ids[i];
  if (env_id < 0 || env_id >= env_count) {
    write_invalid_step(rewards + i, done + i, nagari + i, status + i,
                       primary_committed + i);
    return;
  }

  write_step_result(envs[env_id], actions[i], rewards + i, done + i, nagari + i,
                    status + i, primary_committed + i);
}

std::vector<torch::Tensor> allocate_observation_tensors(
    std::int64_t count,
    const c10::Device& device) {
  auto features = torch::empty(
      {count, static_cast<std::int64_t>(cugo::game::kTorch50FeatureCount)},
      torch::TensorOptions().device(device).dtype(at::kFloat));
  auto legal = torch::empty(
      {count, static_cast<std::int64_t>(cugo::game::kTorch50ActionCount)},
      torch::TensorOptions().device(device).dtype(at::kBool));
  auto players = torch::empty(
      {count}, torch::TensorOptions().device(device).dtype(at::kByte));
  auto done = torch::empty(
      {count}, torch::TensorOptions().device(device).dtype(at::kBool));
  auto status = torch::empty(
      {count}, torch::TensorOptions().device(device).dtype(at::kShort));
  return {features, legal, players, done, status};
}

std::vector<torch::Tensor> allocate_step_tensors(std::int64_t count,
                                                 const c10::Device& device) {
  auto rewards = torch::empty(
      {count}, torch::TensorOptions().device(device).dtype(at::kFloat));
  auto done = torch::empty(
      {count}, torch::TensorOptions().device(device).dtype(at::kBool));
  auto nagari = torch::empty(
      {count}, torch::TensorOptions().device(device).dtype(at::kBool));
  auto status = torch::empty(
      {count}, torch::TensorOptions().device(device).dtype(at::kShort));
  auto committed = torch::empty(
      {count}, torch::TensorOptions().device(device).dtype(at::kBool));
  return {rewards, done, nagari, status, committed};
}

}  // namespace

torch::Tensor cugo_torch50_create_cuda(torch::Tensor seeds,
                                        torch::Tensor first_players) {
  check_cuda_1d(seeds, at::kLong, "seeds");
  check_cuda_1d(first_players, at::kLong, "first_players");
  TORCH_CHECK(seeds.numel() == first_players.numel(),
              "seeds and first_players must have the same length");
  check_same_device(seeds, first_players, "seeds", "first_players");

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
  auto outputs = allocate_observation_tensors(count, states.device());

  if (count != 0) {
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    observe_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const TorchEnv50*>(states.data_ptr<std::uint8_t>()),
        outputs[0].data_ptr<float>(), outputs[1].data_ptr<bool>(),
        outputs[2].data_ptr<std::uint8_t>(), outputs[3].data_ptr<bool>(),
        outputs[4].data_ptr<std::int16_t>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return outputs;
}

std::vector<torch::Tensor> cugo_torch50_observe_indexed_cuda(
    torch::Tensor states,
    torch::Tensor env_ids) {
  check_states(states);
  check_cuda_1d(env_ids, at::kLong, "env_ids");
  check_same_device(states, env_ids, "states", "env_ids");

  const c10::cuda::CUDAGuard guard(states.device());
  const std::int64_t count = env_ids.numel();
  auto outputs = allocate_observation_tensors(count, states.device());

  if (count != 0) {
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    observe_indexed_kernel<<<blocks, kThreads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const TorchEnv50*>(states.data_ptr<std::uint8_t>()),
        env_ids.data_ptr<std::int64_t>(), outputs[0].data_ptr<float>(),
        outputs[1].data_ptr<bool>(), outputs[2].data_ptr<std::uint8_t>(),
        outputs[3].data_ptr<bool>(), outputs[4].data_ptr<std::int16_t>(), count,
        states.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return outputs;
}

std::vector<torch::Tensor> cugo_torch50_step_cuda(torch::Tensor states,
                                                   torch::Tensor actions) {
  check_states(states);
  check_cuda_1d(actions, at::kLong, "actions");
  TORCH_CHECK(states.size(0) == actions.numel(),
              "states and actions must have the same batch size");
  check_same_device(states, actions, "states", "actions");

  const c10::cuda::CUDAGuard guard(states.device());
  const std::int64_t count = states.size(0);
  auto outputs = allocate_step_tensors(count, states.device());

  if (count != 0) {
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    step_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<TorchEnv50*>(states.data_ptr<std::uint8_t>()),
        actions.data_ptr<std::int64_t>(), outputs[0].data_ptr<float>(),
        outputs[1].data_ptr<bool>(), outputs[2].data_ptr<bool>(),
        outputs[3].data_ptr<std::int16_t>(), outputs[4].data_ptr<bool>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return outputs;
}

std::vector<torch::Tensor> cugo_torch50_step_indexed_cuda(
    torch::Tensor states,
    torch::Tensor env_ids,
    torch::Tensor actions) {
  check_states(states);
  check_cuda_1d(env_ids, at::kLong, "env_ids");
  check_cuda_1d(actions, at::kLong, "actions");
  TORCH_CHECK(env_ids.numel() == actions.numel(),
              "env_ids and actions must have the same length");
  check_same_device(states, env_ids, "states", "env_ids");
  check_same_device(states, actions, "states", "actions");

  const c10::cuda::CUDAGuard guard(states.device());
  const std::int64_t count = env_ids.numel();
  auto outputs = allocate_step_tensors(count, states.device());

  if (count != 0) {
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    step_indexed_kernel<<<blocks, kThreads, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<TorchEnv50*>(states.data_ptr<std::uint8_t>()),
        env_ids.data_ptr<std::int64_t>(), actions.data_ptr<std::int64_t>(),
        outputs[0].data_ptr<float>(), outputs[1].data_ptr<bool>(),
        outputs[2].data_ptr<bool>(), outputs[3].data_ptr<std::int16_t>(),
        outputs[4].data_ptr<bool>(), count, states.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return outputs;
}
