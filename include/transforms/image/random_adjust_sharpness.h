#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomAdjustSharpness final : public xt::Module
    {
    public:
        RandomAdjustSharpness();
        explicit RandomAdjustSharpness(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
