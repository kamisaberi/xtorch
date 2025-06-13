#pragma once

#include "common.h"

namespace xt::norm
{
    struct LayerScale : xt::Module
    {
    public:
        LayerScale(int64_t dim, double initial_value = 1e-4);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t dim_; // The feature dimension (number of channels)
        double initial_value_; // Initial value for the learnable scaling factors

        // Learnable scaling parameter (lambda)
        // It's a vector of size 'dim_', one scale factor per channel.
        torch::Tensor lambda_;
    };
}
