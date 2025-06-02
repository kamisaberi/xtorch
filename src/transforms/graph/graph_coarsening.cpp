#include "include/transforms/graph/graph_coarsening.h"

namespace xt::transforms::graph {

    GraphCoarsening::GraphCoarsening() = default;

    GraphCoarsening::GraphCoarsening(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto EdgeDrop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}