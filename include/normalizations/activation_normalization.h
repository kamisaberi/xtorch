#pragma once

#include "common.h"

namespace xt::norm
{
    struct ActiveNorm : xt::Module
    {
    public:
        ActiveNorm(int64_t num_features, double eps = 1e-5, bool affine = true);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Parameters
        int64_t num_features_;
        double eps_;
        bool affine_;

        // Learnable parameters (if affine is true)
        torch::Tensor gamma_; // scale
        torch::Tensor beta_;  // shift

    };
}
