#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomVerticalFlip final : public xt::Module
    {
    public:
        RandomVerticalFlip();
        explicit RandomVerticalFlip(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
