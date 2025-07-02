#pragma once

#include "common.h"

namespace xt::norm
{
    struct GradientNorm : xt::Module
    {
    public:
        GradientNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
