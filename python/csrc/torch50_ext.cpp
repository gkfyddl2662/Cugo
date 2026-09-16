#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "cugo/game/torch50.h"

torch::Tensor cugo_torch50_create_cuda(torch::Tensor seeds,
                                        torch::Tensor first_players);
std::vector<torch::Tensor> cugo_torch50_observe_cuda(torch::Tensor states);
std::vector<torch::Tensor> cugo_torch50_step_cuda(torch::Tensor states,
                                                   torch::Tensor actions);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Cugo Shin Matgo CUDA batch environment";
  m.def("create", &cugo_torch50_create_cuda,
        "Create a GPU-resident batch of TorchEnv50 states");
  m.def("observe", &cugo_torch50_observe_cuda,
        "Encode features and legal masks on the current CUDA stream");
  m.def("step", &cugo_torch50_step_cuda,
        "Apply unified actions to a GPU-resident batch in place");

  m.attr("FEATURE_COUNT") = pybind11::int_(cugo::game::kTorch50FeatureCount);
  m.attr("ACTION_COUNT") = pybind11::int_(cugo::game::kTorch50ActionCount);
  m.attr("PRIMARY_ACTION_COUNT") =
      pybind11::int_(cugo::game::kTorch50PrimaryActionCount);
  m.attr("CHOICE_ACTION_BEGIN") =
      pybind11::int_(cugo::game::kTorch50ChoiceActionBegin);
  m.attr("STATE_BYTES") = pybind11::int_(sizeof(cugo::game::TorchEnv50));
  m.attr("STATUS_OK") =
      pybind11::int_(static_cast<int>(cugo::game::ActionStatus50::kOk));
}
