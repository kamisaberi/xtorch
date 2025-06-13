#pragma once

#include "common.h"

namespace xt::norm
{
    struct EvoNorm : xt::Module
    {
    public:

        EvoNorm(int64_t num_features, int64_t groups = 32, double eps = 1e-5, bool apply_bias = false);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_; // Number of channels C
        int64_t groups_;       // Number of groups to divide channels into for std computation
        double eps_;
        bool apply_bias_;      // Whether to add a learnable bias at the end

        // Learnable parameters
        torch::Tensor v_;      // Per-channel learnable vector 'v'
        torch::Tensor beta_;   // Optional per-channel learnable bias (if apply_bias_ is true)
        // Note: The original paper doesn't explicitly mention a 'gamma' like in BN.
        // 'v' serves a similar scaling role but non-linearly within the sigmoid.

    };
}
