#include "include/transforms/graph/graph_coarsening.h"

namespace xt::transforms::graph {

    GraphCoarsening::GraphCoarsening() = default;

    GraphCoarsening::GraphCoarsening(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto EdgeDrop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}