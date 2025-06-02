#include "include/transforms/graph/edge_add.h"

namespace xt::transforms::graph {

    EdgeAdd::EdgeAdd() = default;

    EdgeAdd::EdgeAdd(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto EdgeDrop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}