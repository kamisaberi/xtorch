#include "include/transforms/graph/feature_augmentation.h"

namespace xt::transforms::graph {

    FeatureAugmentation::FeatureAugmentation() = default;

    FeatureAugmentation::FeatureAugmentation(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto FeatureAugmentation::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}