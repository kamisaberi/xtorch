#pragma once

#include "common.h"

namespace xt::norm
{
    struct SparseSwitchableNorm : xt::Module
    {
    public:
        SparseSwitchableNorm(int64_t num_features,
                             double eps_bn = 1e-5, double momentum_bn = 0.1, bool affine_bn = true,
                             double eps_in = 1e-5, bool affine_in = true,
                             double eps_ln = 1e-5, bool affine_ln = true);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_;
        static const int kNumNormalizers = 3; // BN, IN, LN

        // BN parameters
        double eps_bn_;
        double momentum_bn_;
        bool affine_bn_;
        torch::Tensor running_mean_bn_;
        torch::Tensor running_var_bn_;
        torch::Tensor gamma_bn_;
        torch::Tensor beta_bn_;
        torch::Tensor num_batches_tracked_bn_;

        // IN parameters
        double eps_in_;
        bool affine_in_;
        torch::Tensor gamma_in_;
        torch::Tensor beta_in_;

        // LN parameters
        // LayerNorm normalizes over the last D dimensions. For typical (N,C,H,W),
        // and if we want to normalize over C,H,W, normalized_shape_ln_ would be {C,H,W}.
        // This makes LN params dependent on H,W.
        // A common simplification for SN is to have LN normalize only over C (like GroupNorm(1,C))
        // or to require fixed H,W for LN's specific affine params.
        // For this example, let's make LN normalize over C,H,W. We'll need to pass H,W or initialize LN later.
        // Or, use functional layer_norm and manage its affine params.
        // Let's make LN params dependent on H,W (initialized on first forward).
        std::vector<int64_t> normalized_shape_ln_; // e.g., {C,H,W}
        double eps_ln_;
        bool affine_ln_;
        torch::nn::LayerNorm layer_norm_{nullptr}; // To be initialized
        // Alternatively, manage gamma_ln, beta_ln manually if using functional layer_norm.
        // For simplicity with LayerNorm module, we defer its full init or assume it normalizes C.

        // Learnable importance scores (logits) for mixing normalizers
        // Shape: (num_features, kNumNormalizers) -> one set of scores per channel
        torch::Tensor mixing_logits_;
    };
}
