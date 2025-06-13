#pragma once

#include "common.h"

namespace xt::dropouts
{
    torch::Tensor band_dropout(torch::Tensor x);

    struct BandDropout : xt::Module
    {
    public:
        BandDropout(c10::IntArrayRef params_shape = {},
                    double initial_baseline_dropout_rate = 0.1,
                    double initial_alpha_value = 0.01);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        torch::Tensor alpha_;
        torch::Tensor beta_;
        double epsilon_ = 1e-7; // For numerical stability
    };
}
