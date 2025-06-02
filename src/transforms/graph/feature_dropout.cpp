#include "include/transforms/graph/feature_dropout.h"

namespace xt::transforms::graph {

    FeatureDropout::FeatureDropout() = default;

    FeatureDropout::FeatureDropout(std::vector<xt::Module> transforms) : xt::Module(), transform(transform) {
    }

    auto FeatureDropout::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}