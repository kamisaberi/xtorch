#include "include/transforms/graph/random_walk_subgraph.h"

namespace xt::transforms::graph {

    RandomWalkSubgraph::RandomWalkSubgraph() = default;

    RandomWalkSubgraph::RandomWalkSubgraph(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto RandomWalkSubgraph::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}