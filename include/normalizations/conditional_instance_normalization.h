#pragma once

#include "common.h"

namespace xt::norm
{
    struct ConditionalInstanceNorm : xt::Module
    {
    public:
        ConditionalInstanceNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
