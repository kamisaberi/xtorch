#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomAugment final : public xt::Module
    {
    public:
        RandomAugment();
        explicit RandomAugment(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
