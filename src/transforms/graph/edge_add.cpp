#include "include/transforms/graph/edge_add.h"

namespace xt::transforms::graph {

    EdgeAdd::EdgeAdd() = default;

    EdgeAdd::EdgeAdd(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto EdgeDrop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}