#pragma once

#include "common.h"

namespace xt::norm
{
    struct PixelNorm : xt::Module
    {
    public:
        PixelNorm(double eps = 1e-8);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double eps_; // Small epsilon for numerical stability
    };
}
