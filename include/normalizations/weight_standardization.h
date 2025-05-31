#pragma once

#include "common.h"

namespace xt::norm
{
    struct WeightStandardization : xt::Module
    {
    public:
        WeightStandardization() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
