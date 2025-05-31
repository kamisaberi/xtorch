#include "include/transforms/graph/node_feature_masking.h"

namespace xt::transforms::graph {

    NodeFeatureMasking::NodeFeatureMasking() = default;

    NodeFeatureMasking::NodeFeatureMasking(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto NodeFeatureMasking::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}