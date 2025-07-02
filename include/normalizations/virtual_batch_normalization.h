#pragma once

#include "common.h"

namespace xt::norm
{
    struct VirtualBatchNorm : xt::Module
    {
    public:
        VirtualBatchNorm(int64_t num_features, double eps = 1e-5, bool affine = true);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_;
        double eps_;
        bool affine_; // Whether to apply learnable affine transform

        // Learnable affine parameters (gamma and beta)
        torch::Tensor gamma_;
        torch::Tensor beta_;

        // Buffers for storing reference statistics (mu_ref, var_ref)
        // These are computed once from a reference batch.
        torch::Tensor mu_ref_;
        torch::Tensor var_ref_;
        torch::Tensor initialized_flag_; // To check if reference stats have been set
    };
}
