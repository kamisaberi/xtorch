#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class FeatureDropout final : public xt::Module
    {
    public:
        FeatureDropout();
        explicit FeatureDropout(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
