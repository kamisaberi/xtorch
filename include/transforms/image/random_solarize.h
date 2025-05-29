#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomSolarize final : public xt::Module
    {
    public:
        RandomSolarize();
        explicit RandomSolarize(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
