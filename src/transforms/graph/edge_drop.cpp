#include "include/transforms/graph/edge_drop.h"

namespace xt::transforms::graph {

    EdgeDrop::EdgeDrop() = default;

    EdgeDrop::EdgeDrop(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto EdgeDrop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}