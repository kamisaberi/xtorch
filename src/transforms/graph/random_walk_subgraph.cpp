#include "include/transforms/graph/random_walk_subgraph.h"

namespace xt::transforms::graph {

    RandomWalkSubgraph::RandomWalkSubgraph() = default;

    RandomWalkSubgraph::RandomWalkSubgraph(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto RandomWalkSubgraph::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}