#pragma once

#include "common.h"

namespace xt::norm
{
    struct PixelNorm : xt::Module
    {
    public:
        PixelNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
