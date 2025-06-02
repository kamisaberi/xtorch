#include "include/transforms/graph/node_feature_shuffling.h"

namespace xt::transforms::graph {

    NodeFeatureShuffling::NodeFeatureShuffling() = default;

    NodeFeatureShuffling::NodeFeatureShuffling(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto NodeFeatureShuffling::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}