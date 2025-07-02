#pragma once

#include "common.h"

namespace xt::norm
{
    struct AdaptiveInstanceNorm : xt::Module
    {
    public:
        explicit AdaptiveInstanceNorm(double eps = 1e-5);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double eps_;
    };
}
