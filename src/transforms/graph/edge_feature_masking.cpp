#include "include/transforms/graph/edge_feature_masking.h"

namespace xt::transforms::graph {

    EdgeFeatureMasking::EdgeFeatureMasking() = default;

    EdgeFeatureMasking::EdgeFeatureMasking(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto EdgeFeatureMasking::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}