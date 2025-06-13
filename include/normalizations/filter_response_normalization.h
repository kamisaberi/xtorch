#pragma once

#include "common.h"

namespace xt::norm
{
    struct FilterResponseNorm : xt::Module
    {
    public:

        FilterResponseNorm(int64_t num_features, double eps = 1e-6);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_; // Number of channels C
        double eps_; // Small epsilon for numerical stability in FRN

        // Learnable parameters for FRN (affine transformation)
        torch::Tensor gamma_; // Per-channel scale
        torch::Tensor beta_; // Per-channel shift

        // Learnable parameter for TLU (Thresholded Linear Unit)
        torch::Tensor tau_; // Per-channel threshold
    };
}
