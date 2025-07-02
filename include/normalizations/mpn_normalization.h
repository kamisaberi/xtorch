#pragma once

#include "common.h"

namespace xt::norm
{
    struct MPNNorm : xt::Module
    {
    public:
        MPNNorm(const std::vector<int64_t>& normalized_shape, // e.g., {num_features}
                double eps = 1e-5,
                bool elementwise_affine = true);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // This implementation will use Layer Normalization.
        // The 'normalized_shape' for LayerNorm will be the feature dimension.
        std::vector<int64_t> normalized_shape_; // Should be {num_features}
        double eps_;
        bool elementwise_affine_; // Whether LayerNorm has learnable gamma and beta

        // Layer Normalization components
        torch::Tensor gamma_; // Scale (if elementwise_affine is true)
        torch::Tensor beta_; // Shift (if elementwise_affine is true)
    };
}
