#pragma once

#include "common.h"

namespace xt::norm
{
    struct WeightDemodulization : xt::Module
    {
    public:
        WeightDemodulization() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
