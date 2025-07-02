#pragma once

#include "common.h"

namespace xt::norm
{
    struct CosineNorm : xt::Module
    {
    public:
        CosineNorm(int64_t dim = 1, double eps = 1e-8, bool learnable_tau = false, double initial_tau = 1.0);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t dim_;       // Dimension along which to normalize (typically the feature dimension)
        double eps_;        // Small epsilon to prevent division by zero
        bool learnable_tau_; // Whether to include a learnable temperature parameter
        torch::Tensor tau_;  // Learnable temperature parameter (scalar)

    };
}
