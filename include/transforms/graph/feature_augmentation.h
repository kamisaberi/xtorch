#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::graph
{
    class FeatureAugmentation final : public xt::Module
    {
    public:
        FeatureAugmentation();
        explicit FeatureAugmentation(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
