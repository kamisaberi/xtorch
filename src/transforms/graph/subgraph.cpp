#include "include/transforms/graph/subgraph.h"

namespace xt::transforms::graph {

    Subgraph::Subgraph() = default;

    Subgraph::Subgraph(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto Subgraph::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}