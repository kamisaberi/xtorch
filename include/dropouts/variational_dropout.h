#pragma once

#include "common.h"

namespace xt::dropouts
{
    struct VariationalDropout : xt::Module
    {
    public:
        VariationalDropout(c10::IntArrayRef alpha_shape = {}, double initial_dropout_rate_for_alpha_init = 0.05);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        torch::Tensor log_alpha_;
        double epsilon_ = 1e-8; // For numerical stability, esp. in sqrt
    };
}
