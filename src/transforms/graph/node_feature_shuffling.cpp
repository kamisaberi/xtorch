#include "include/transforms/graph/node_feature_shuffling.h"

namespace xt::transforms::graph {

    NodeFeatureShuffling::NodeFeatureShuffling() = default;

    NodeFeatureShuffling::NodeFeatureShuffling(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto NodeFeatureShuffling::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}