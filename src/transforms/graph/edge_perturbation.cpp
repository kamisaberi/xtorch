#include "include/transforms/graph/edge_perturbation.h"

namespace xt::transforms::graph {

    EdgePerturbation::EdgePerturbation() = default;

    EdgePerturbation::EdgePerturbation(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto EdgePerturbation::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}