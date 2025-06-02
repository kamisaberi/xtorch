#include "include/transforms/graph/edge_perturbation.h"

namespace xt::transforms::graph {

    EdgePerturbation::EdgePerturbation() = default;

    EdgePerturbation::EdgePerturbation(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto EdgePerturbation::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}