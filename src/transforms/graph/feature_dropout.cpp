#include "include/transforms/graph/feature_dropout.h"

namespace xt::transforms::graph {

    FeatureDropout::FeatureDropout() = default;

    FeatureDropout::FeatureDropout(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto FeatureDropout::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}