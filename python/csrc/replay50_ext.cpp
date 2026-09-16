#include <torch/extension.h>

#include <cstdint>
#include <vector>

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
    torch::Tensor targets);

std::vector<torch::Tensor> cugo_replay50_gather_unpack_cuda(
    torch::Tensor feature_bits,
    torch::Tensor feature_scalars,
    torch::Tensor legal_bits,
    torch::Tensor packed_actions,
    torch::Tensor packed_targets,
    torch::Tensor index);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Cugo packed Torch50 replay CUDA kernels";
  m.def("pack_into", &cugo_replay50_pack_into_cuda,
        "Pack Torch50 replay rows directly into preallocated storage");
  m.def("gather_unpack", &cugo_replay50_gather_unpack_cuda,
        "Gather packed replay rows and unpack one training minibatch");

  m.attr("FEATURE_COUNT") = pybind11::int_(496);
  m.attr("FEATURE_BINARY_COUNT") = pybind11::int_(453);
  m.attr("FEATURE_SCALAR_COUNT") = pybind11::int_(26);
  m.attr("FEATURE_PACK_BYTES") = pybind11::int_(57);
  m.attr("ACTION_COUNT") = pybind11::int_(177);
  m.attr("LEGAL_PACK_BYTES") = pybind11::int_(23);
}
