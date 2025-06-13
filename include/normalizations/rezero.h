#pragma once

#include "common.h"

namespace xt::norm
{
    struct ReZero : xt::Module
    {
    public:
        ReZero(double initial_alpha_value = 0.0);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Learnable parameter 'alpha'
        // Initialized to zero, as per the ReZero paper.
        torch::Tensor alpha_;

    };
}
