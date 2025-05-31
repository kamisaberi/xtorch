#pragma once

#include "common.h"

namespace xt::norm
{
    struct AdaptiveInstanceNorm : xt::Module
    {
    public:
        AdaptiveInstanceNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
