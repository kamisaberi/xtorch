#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomGridShuffle final : public xt::Module
    {
    public:
        RandomGridShuffle();
        explicit RandomGridShuffle(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
