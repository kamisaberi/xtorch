#include "include/transforms/graph/feature_augmentation.h"

namespace xt::transforms::graph {

    FeatureAugmentation::FeatureAugmentation() = default;

    FeatureAugmentation::FeatureAugmentation(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto FeatureAugmentation::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}