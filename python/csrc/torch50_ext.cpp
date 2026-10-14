#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "cugo/game/torch50.h"

torch::Tensor cugo_torch50_create_cuda(torch::Tensor seeds,
                                        torch::Tensor first_players);
std::vector<torch::Tensor> cugo_torch50_observe_cuda(torch::Tensor states);
std::vector<torch::Tensor> cugo_torch50_observe_indexed_cuda(
    torch::Tensor states,
    torch::Tensor env_ids);
std::vector<torch::Tensor> cugo_torch50_step_cuda(torch::Tensor states,
                                                   torch::Tensor actions);
std::vector<torch::Tensor> cugo_torch50_step_indexed_cuda(
    torch::Tensor states,
    torch::Tensor env_ids,
    torch::Tensor actions);
torch::Tensor cugo_torch50_clone_indexed_cuda(torch::Tensor states,
                                               torch::Tensor env_ids);
std::vector<torch::Tensor> cugo_torch50_branch_indexed_cuda(
    torch::Tensor states,
    torch::Tensor env_ids,
    torch::Tensor actions);
torch::Tensor cugo_torch50_hands_indexed_cuda(torch::Tensor states,
                                              torch::Tensor env_ids);
torch::Tensor cugo_torch50_stock_indexed_cuda(torch::Tensor states,
                                              torch::Tensor env_ids);
std::vector<torch::Tensor> cugo_torch50_step_events_indexed_cuda(
    torch::Tensor states,
    torch::Tensor env_ids,
    torch::Tensor actions);
torch::Tensor cugo_torch50_start_cuda(torch::Tensor states);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Cugo Shin Matgo CUDA batch environment";
  m.def("create", &cugo_torch50_create_cuda,
        "Create a GPU-resident batch of TorchEnv50 states");
  m.def("observe", &cugo_torch50_observe_cuda,
        "Encode features and legal masks on the current CUDA stream");
  m.def("observe_indexed", &cugo_torch50_observe_indexed_cuda,
        "Encode only selected environment rows on the current CUDA stream");
  m.def("step", &cugo_torch50_step_cuda,
        "Apply unified actions to a GPU-resident batch in place");
  m.def("step_indexed", &cugo_torch50_step_indexed_cuda,
        "Apply unified actions only to selected environment rows in place");
  m.def("clone_indexed", &cugo_torch50_clone_indexed_cuda,
        "Clone selected environment rows on the current CUDA stream");
  m.def("branch_indexed", &cugo_torch50_branch_indexed_cuda,
        "Clone selected rows and apply actions without mutating parents");
  m.def("hands_indexed", &cugo_torch50_hands_indexed_cuda,
        "Return [N, 2] int64 card masks of both players' hidden hands");
  m.def("stock_indexed", &cugo_torch50_stock_indexed_cuda,
        "Return [N] int64: the next 8 stock draws, one card id per byte (0xFF = none)");
  m.def("step_events_indexed", &cugo_torch50_step_events_indexed_cuda,
        "step_indexed that also returns [N, EVENT_WORDS] int64 event records");
  m.def("start", &cugo_torch50_start_cuda,
        "Return [N, START_WORDS] int64 opening records of fresh games");

  m.attr("FEATURE_COUNT") = pybind11::int_(cugo::game::kTorch50FeatureCount);
  m.attr("ACTION_COUNT") = pybind11::int_(cugo::game::kTorch50ActionCount);
  m.attr("PRIMARY_ACTION_COUNT") =
      pybind11::int_(cugo::game::kTorch50PrimaryActionCount);
  m.attr("CHOICE_ACTION_BEGIN") =
      pybind11::int_(cugo::game::kTorch50ChoiceActionBegin);
  m.attr("EVENT_WORDS") = pybind11::int_(cugo::game::kTorch50EventWords);
  m.attr("START_WORDS") = pybind11::int_(cugo::game::kTorch50StartWords);
  m.attr("STATE_BYTES") = pybind11::int_(sizeof(cugo::game::TorchEnv50));
  m.attr("STATUS_OK") =
      pybind11::int_(static_cast<int>(cugo::game::ActionStatus50::kOk));
}
