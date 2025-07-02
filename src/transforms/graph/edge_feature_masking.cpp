#include "include/transforms/graph/edge_feature_masking.h"

namespace xt::transforms::graph {

    EdgeFeatureMasking::EdgeFeatureMasking() = default;

    EdgeFeatureMasking::EdgeFeatureMasking(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto EdgeFeatureMasking::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}