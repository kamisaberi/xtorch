#pragma once

#include "common.h"
#include <torch/torch.h>
#include <vector>
#include <cmath>     // For std::log
#include <ostream>   // For std::ostream


namespace
{
    // Anonymous namespace for helper utility
    double calculate_initial_log_alpha_value(double initial_dropout_rate)
    {
        // Clamp initial_dropout_rate to avoid log(0) or log( división by zero )
        double epsilon = 1e-7;
        if (initial_dropout_rate < epsilon)
        {
            initial_dropout_rate = epsilon;
        }
        if (initial_dropout_rate > 1.0 - epsilon)
        {
            initial_dropout_rate = 1.0 - epsilon;
        }
        return std::log(initial_dropout_rate / (1.0 - initial_dropout_rate));
    }
}

namespace xt::dropouts
{
    torch::Tensor adaptive_dropout(torch::Tensor x);

    struct AdaptiveDropout : xt::Module
    {
    public:
        explicit AdaptiveDropout(c10::IntArrayRef probability_shape = {}, double initial_dropout_rate = 0.05);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        torch::Tensor log_alpha_;
    };
}
