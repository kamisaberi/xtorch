#include "include/transforms/graph/node_drop.h"

namespace xt::transforms::graph {

    NodeDrop::NodeDrop() = default;

    NodeDrop::NodeDrop(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto NodeDrop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}