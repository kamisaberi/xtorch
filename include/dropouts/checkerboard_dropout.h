#pragma once

#include "common.h"

namespace xt::dropouts
{
    torch::Tensor checkerboard_dropout(torch::Tensor x);

    struct CheckerboardDropout : xt::Module
    {
    public:
        CheckerboardDropout(bool drop_even_sum_indices = true);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        bool drop_even_sum_indices_; // If true, drop elements where sum of relevant indices is even.
        // If false, drop elements where sum of relevant indices is odd.
    };
}
