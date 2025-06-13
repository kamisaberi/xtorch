#pragma once

#include "common.h"

namespace xt::dropouts
{
    torch::Tensor concrete_dropout(torch::Tensor x);

    struct ConcreteDropout : xt::Module
    {
    public:
        ConcreteDropout(
            c10::IntArrayRef probability_shape = {}, // Shape of log_alpha (empty for scalar)
            double initial_dropout_rate = 0.05, // Desired initial dropout probability p
            double temperature = 0.1, // Temperature for Concrete distribution
            double dropout_regularizer = 1e-5); // Multiplier for the regularization term
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        torch::Tensor log_alpha_; // Unconstrained learnable parameter(s) for dropout probability
        double temperature_;
        double dropout_regularizer_factor_; // Factor to scale the p-based regularization term
        double epsilon_ = 1e-7; // Small constant for numerical stability
    };
}
