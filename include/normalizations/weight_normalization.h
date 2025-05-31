#pragma once

#include "common.h"

namespace xt::norm
{
    struct WeightNorm : xt::Module
    {
    public:
        WeightNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
