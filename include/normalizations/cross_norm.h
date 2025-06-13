#pragma once

#include "common.h"

namespace xt::norm
{
    struct CrossNorm : xt::Module
    {
    public:
        CrossNorm(int64_t num_features, double eps = 1e-5, bool affine = true);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_;
        double eps_;
        bool affine_; // Whether to apply learnable affine transform to x after normalization

        // Learnable parameters for x (if affine is true)
        torch::Tensor gamma_x_; // scale for x
        torch::Tensor beta_x_;  // shift for x

    };
}
