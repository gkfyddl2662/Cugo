#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kFeatureCount = 496;
constexpr int kFeatureBinaryCount = 453;
constexpr int kFeatureScalarOffset = 453;
constexpr int kFeatureScalarCount = 26;
constexpr int kFeaturePackBytes = 57;
constexpr int kActionCount = 177;
constexpr int kLegalPackBytes = 23;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;

void check_cuda(const torch::Tensor& tensor,
                at::ScalarType dtype,
                const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

void check_1d(const torch::Tensor& tensor,
              at::ScalarType dtype,
              const char* name) {
  check_cuda(tensor, dtype, name);
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
}

void check_2d(const torch::Tensor& tensor,
              at::ScalarType dtype,
              std::int64_t width,
              const char* name) {
  check_cuda(tensor, dtype, name);
  TORCH_CHECK(tensor.dim() == 2, name, " must be two-dimensional");
  TORCH_CHECK(tensor.size(1) == width, name, " has the wrong second dimension");
}

void check_same_device(const torch::Tensor& first,
                       const torch::Tensor& second,
                       const char* first_name,
                       const char* second_name) {
  TORCH_CHECK(first.device() == second.device(), first_name, " and ",
              second_name, " must be on the same CUDA device");
}

__global__ void pack_into_kernel(
    std::uint8_t* feature_bits,
    __half* feature_scalars,
    std::uint8_t* legal_bits,
    std::uint8_t* packed_actions,
    float* packed_targets,
    std::int64_t dst_start,
    const __half* features,
    const bool* legal,
    const std::int64_t* actions,
    const float* targets,
    std::int64_t count) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::int64_t warp =
      (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kWarpSize;
  if (warp >= count) return;

  const std::int64_t src_row = warp;
  const std::int64_t dst_row = dst_start + warp;
  const __half* feature_row = features + src_row * kFeatureCount;
  const bool* legal_row = legal + src_row * kActionCount;

  for (int byte = lane; byte < kFeaturePackBytes; byte += kWarpSize) {
    std::uint8_t value = 0;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const int feature = byte * 8 + bit;
      if (feature < kFeatureBinaryCount &&
          __half2float(feature_row[feature]) != 0.0f) {
        value = static_cast<std::uint8_t>(value | (1u << bit));
      }
    }
    feature_bits[dst_row * kFeaturePackBytes + byte] = value;
  }

  for (int scalar = lane; scalar < kFeatureScalarCount; scalar += kWarpSize) {
    feature_scalars[dst_row * kFeatureScalarCount + scalar] =
        feature_row[kFeatureScalarOffset + scalar];
  }

  for (int byte = lane; byte < kLegalPackBytes; byte += kWarpSize) {
    std::uint8_t value = 0;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const int action = byte * 8 + bit;
      if (action < kActionCount && legal_row[action]) {
        value = static_cast<std::uint8_t>(value | (1u << bit));
      }
    }
    legal_bits[dst_row * kLegalPackBytes + byte] = value;
  }

  if (lane == 0) {
    packed_actions[dst_row] = static_cast<std::uint8_t>(actions[src_row]);
    packed_targets[dst_row] = targets[src_row];
  }
}

__global__ void gather_unpack_kernel(
    const std::uint8_t* feature_bits,
    const __half* feature_scalars,
    const std::uint8_t* legal_bits,
    const std::uint8_t* packed_actions,
    const float* packed_targets,
    const std::int64_t* index,
    __half* features,
    bool* legal,
    std::int64_t* actions,
    float* targets,
    std::int64_t count) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::int64_t warp =
      (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kWarpSize;
  if (warp >= count) return;

  const std::int64_t src_row = index[warp];
  __half* feature_row = features + warp * kFeatureCount;
  bool* legal_row = legal + warp * kActionCount;

  for (int feature = lane; feature < kFeatureCount; feature += kWarpSize) {
    if (feature < kFeatureBinaryCount) {
      const std::uint8_t byte =
          feature_bits[src_row * kFeaturePackBytes + feature / 8];
      const float value = static_cast<float>((byte >> (feature & 7)) & 1u);
      feature_row[feature] = __float2half(value);
    } else if (feature < kFeatureScalarOffset + kFeatureScalarCount) {
      feature_row[feature] = feature_scalars[
          src_row * kFeatureScalarCount + feature - kFeatureScalarOffset];
    } else {
      feature_row[feature] = __float2half(0.0f);
    }
  }

  for (int action = lane; action < kActionCount; action += kWarpSize) {
    const std::uint8_t byte =
        legal_bits[src_row * kLegalPackBytes + action / 8];
    legal_row[action] = ((byte >> (action & 7)) & 1u) != 0;
  }

  if (lane == 0) {
    actions[warp] = static_cast<std::int64_t>(packed_actions[src_row]);
    targets[warp] = packed_targets[src_row];
  }
}

void check_storage(
    const torch::Tensor& feature_bits,
    const torch::Tensor& feature_scalars,
    const torch::Tensor& legal_bits,
    const torch::Tensor& packed_actions,
    const torch::Tensor& packed_targets) {
  check_2d(feature_bits, at::kByte, kFeaturePackBytes, "feature_bits");
  check_2d(feature_scalars, at::kHalf, kFeatureScalarCount,
           "feature_scalars");
  check_2d(legal_bits, at::kByte, kLegalPackBytes, "legal_bits");
  check_1d(packed_actions, at::kByte, "packed_actions");
  check_1d(packed_targets, at::kFloat, "packed_targets");

  const std::int64_t capacity = feature_bits.size(0);
  TORCH_CHECK(feature_scalars.size(0) == capacity,
              "feature_scalars capacity mismatch");
  TORCH_CHECK(legal_bits.size(0) == capacity,
              "legal_bits capacity mismatch");
  TORCH_CHECK(packed_actions.numel() == capacity,
              "packed_actions capacity mismatch");
  TORCH_CHECK(packed_targets.numel() == capacity,
              "packed_targets capacity mismatch");

  check_same_device(feature_bits, feature_scalars, "feature_bits",
                    "feature_scalars");
  check_same_device(feature_bits, legal_bits, "feature_bits", "legal_bits");
  check_same_device(feature_bits, packed_actions, "feature_bits",
                    "packed_actions");
  check_same_device(feature_bits, packed_targets, "feature_bits",
                    "packed_targets");
}

}  // namespace

void cugo_replay50_pack_into_cuda(
    torch::Tensor feature_bits,
    torch::Tensor feature_scalars,
    torch::Tensor legal_bits,
    torch::Tensor packed_actions,
    torch::Tensor packed_targets,
    std::int64_t dst_start,
    torch::Tensor features,
    torch::Tensor legal,
    torch::Tensor actions,
    torch::Tensor targets) {
  check_storage(feature_bits, feature_scalars, legal_bits, packed_actions,
                packed_targets);
  check_2d(features, at::kHalf, kFeatureCount, "features");
  check_2d(legal, at::kBool, kActionCount, "legal");
  check_1d(actions, at::kLong, "actions");
  check_1d(targets, at::kFloat, "targets");

  const std::int64_t count = features.size(0);
  TORCH_CHECK(legal.size(0) == count, "features and legal row count mismatch");
  TORCH_CHECK(actions.numel() == count,
              "features and actions row count mismatch");
  TORCH_CHECK(targets.numel() == count,
              "features and targets row count mismatch");
  TORCH_CHECK(dst_start >= 0, "dst_start must be non-negative");
  TORCH_CHECK(dst_start + count <= feature_bits.size(0),
              "packed replay write exceeds storage capacity");

  check_same_device(feature_bits, features, "feature_bits", "features");
  check_same_device(feature_bits, legal, "feature_bits", "legal");
  check_same_device(feature_bits, actions, "feature_bits", "actions");
  check_same_device(feature_bits, targets, "feature_bits", "targets");

  const c10::cuda::CUDAGuard guard(feature_bits.device());
  if (count == 0) return;

  const int blocks =
      static_cast<int>((count + kWarpsPerBlock - 1) / kWarpsPerBlock);
  pack_into_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      feature_bits.data_ptr<std::uint8_t>(),
      reinterpret_cast<__half*>(feature_scalars.data_ptr<at::Half>()),
      legal_bits.data_ptr<std::uint8_t>(),
      packed_actions.data_ptr<std::uint8_t>(), packed_targets.data_ptr<float>(),
      dst_start, reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
      legal.data_ptr<bool>(), actions.data_ptr<std::int64_t>(),
      targets.data_ptr<float>(), count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> cugo_replay50_gather_unpack_cuda(
    torch::Tensor feature_bits,
    torch::Tensor feature_scalars,
    torch::Tensor legal_bits,
    torch::Tensor packed_actions,
    torch::Tensor packed_targets,
    torch::Tensor index) {
  check_storage(feature_bits, feature_scalars, legal_bits, packed_actions,
                packed_targets);
  check_1d(index, at::kLong, "index");
  check_same_device(feature_bits, index, "feature_bits", "index");

  const c10::cuda::CUDAGuard guard(feature_bits.device());
  const std::int64_t count = index.numel();
  auto features = torch::empty(
      {count, kFeatureCount},
      torch::TensorOptions().device(feature_bits.device()).dtype(at::kHalf));
  auto legal = torch::empty(
      {count, kActionCount},
      torch::TensorOptions().device(feature_bits.device()).dtype(at::kBool));
  auto actions = torch::empty(
      {count},
      torch::TensorOptions().device(feature_bits.device()).dtype(at::kLong));
  auto targets = torch::empty(
      {count},
      torch::TensorOptions().device(feature_bits.device()).dtype(at::kFloat));

  if (count != 0) {
    const int blocks =
        static_cast<int>((count + kWarpsPerBlock - 1) / kWarpsPerBlock);
    gather_unpack_kernel<<<blocks, kThreads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        feature_bits.data_ptr<std::uint8_t>(),
        reinterpret_cast<const __half*>(feature_scalars.data_ptr<at::Half>()),
        legal_bits.data_ptr<std::uint8_t>(),
        packed_actions.data_ptr<std::uint8_t>(), packed_targets.data_ptr<float>(),
        index.data_ptr<std::int64_t>(),
        reinterpret_cast<__half*>(features.data_ptr<at::Half>()),
        legal.data_ptr<bool>(), actions.data_ptr<std::int64_t>(),
        targets.data_ptr<float>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {features, legal, actions, targets};
}
