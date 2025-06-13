#pragma once

#include "common.h"

namespace xt::norm
{
    struct DecorrelatedBatchNorm : xt::Module
    {
    public:
        DecorrelatedBatchNorm(int64_t num_features,
                                           double eps_bn = 1e-5,
                                           double momentum_bn = 0.1,
                                           bool affine_bn = false, // Often false or fixed for DBN's BN part
                                           bool affine_final = true);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_; // Number of input features (e.g., C if input is N,C or C*H*W if flattened)
        double eps_bn_;        // Epsilon for the initial Batch Normalization part
        double momentum_bn_;   // Momentum for the initial Batch Normalization part
        bool affine_bn_;       // Whether the initial BN has affine parameters (gamma, beta)
        // For DBN, often the affine params of initial BN are disabled or fixed
        // as the decorrelation matrix and final affine params handle scaling/shifting.
        bool affine_final_;    // Whether to apply final affine parameters after decorrelation

        // Standard Batch Normalization components
        torch::Tensor running_mean_bn_;
        torch::Tensor running_var_bn_;
        torch::Tensor gamma_bn_; // Optional affine for BN part
        torch::Tensor beta_bn_;  // Optional affine for BN part
        torch::Tensor num_batches_tracked_bn_;

        // Decorrelation components
        // We learn a linear transformation W for decorrelation.
        // W should be (num_features_, num_features_)
        torch::Tensor decorrelation_matrix_W_; // W

        // Final affine parameters (optional, applied after decorrelation)
        torch::Tensor gamma_final_;
        torch::Tensor beta_final_;

        // IterNorm specific (for more advanced decorrelation, not fully implemented here for simplicity)
        int num_iter_newton_; // Number of Newton's iterations for Sigma^{-1/2}

    };
}
