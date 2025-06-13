#pragma once

#include "common.h"
#include <torch/torch.h>
#include <vector>
#include <cmath>     // For std::log
#include <ostream>   // For std::ostream


namespace xt::dropouts
{
    namespace
    {
        double calculate_initial_log_alpha_value(double initial_dropout_rate);
    }


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
