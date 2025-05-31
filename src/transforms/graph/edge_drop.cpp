#include "include/transforms/graph/edge_drop.h"

namespace xt::transforms::graph {

    EdgeDrop::EdgeDrop() = default;

    EdgeDrop::EdgeDrop(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto EdgeDrop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}