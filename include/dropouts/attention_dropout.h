#pragma once

#include "common.h"

namespace xt::dropouts
{
    struct AttentionDropout : xt::Module
    {
    public:
        AttentionDropout(double p = 0.1);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_; // Probability of an element to be zeroed.
    };
}
