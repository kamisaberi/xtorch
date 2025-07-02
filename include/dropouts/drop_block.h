#pragma once

#include "common.h"

namespace xt::dropouts
{
    struct DropBlock : xt::Module
    {
    public:
        DropBlock(int block_size = 7, double drop_prob = 0.1);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int block_size_;
        double drop_prob_; // This is the target probability 'p' for dropping units, similar to standard Dropout
        double epsilon_ = 1e-6; // For numerical stability in division
    };
}
