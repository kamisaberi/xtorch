#include "include/transforms/graph/subgraph.h"

namespace xt::transforms::graph {

    Subgraph::Subgraph() = default;

    Subgraph::Subgraph(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto Subgraph::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}