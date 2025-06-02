#include "include/transforms/graph/node_feature_masking.h"

namespace xt::transforms::graph {

    NodeFeatureMasking::NodeFeatureMasking() = default;

    NodeFeatureMasking::NodeFeatureMasking(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto NodeFeatureMasking::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}