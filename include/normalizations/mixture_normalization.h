#pragma once

#include "common.h"

namespace xt::norm
{
    struct MixtureNorm : xt::Module
    {
    public:
        MixtureNorm(int64_t num_features,
                    double eps_bn = 1e-5, double momentum_bn = 0.1, bool affine_bn = true,
                    double eps_in = 1e-5, bool affine_in = true);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_;
        double eps_bn_;
        double momentum_bn_;
        double eps_in_;
        bool affine_bn_; // Whether the BN component has its own affine parameters
        bool affine_in_; // Whether the IN component has its own affine parameters
        // Note: The final output will be a mix. If both are true, it's a mix of affined outputs.
        // One could also have a single final affine transformation after mixing. For simplicity,
        // let's assume the sub-normalizers handle their own affine if enabled.

        // Batch Normalization components
        torch::Tensor running_mean_bn_;
        torch::Tensor running_var_bn_;
        torch::Tensor gamma_bn_;
        torch::Tensor beta_bn_;
        torch::Tensor num_batches_tracked_bn_;

        // Instance Normalization components (no running stats, only optional affine)
        torch::Tensor gamma_in_;
        torch::Tensor beta_in_;

        // Learnable mixing parameters (lambda)
        // We'll learn `lambda_raw_` and pass it through sigmoid to keep it in [0, 1]
        torch::Tensor lambda_raw_; // Per-channel mixing weights
    };
}
