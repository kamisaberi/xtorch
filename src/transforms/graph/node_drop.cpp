#include "include/transforms/graph/node_drop.h"

namespace xt::transforms::graph {

    NodeDrop::NodeDrop() = default;

    NodeDrop::NodeDrop(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto NodeDrop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}