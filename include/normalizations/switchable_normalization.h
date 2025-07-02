#pragma once

#include "common.h"

namespace xt::norm
{
    struct SwitchableNorm : xt::Module
    {
    public:
        SwitchableNorm(int64_t num_features,
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
        bool affine_bn_; // If true, BN has its own gamma_bn, beta_bn
        torch::Tensor running_mean_bn_;
        torch::Tensor running_var_bn_;
        torch::Tensor gamma_bn_;
        torch::Tensor beta_bn_;
        torch::Tensor num_batches_tracked_bn_;

        // IN parameters
        double eps_in_;
        bool affine_in_; // If true, IN has its own gamma_in, beta_in
        torch::Tensor gamma_in_;
        torch::Tensor beta_in_;

        // LN parameters
        std::vector<int64_t> normalized_shape_ln_; // e.g., {C,H,W}
        double eps_ln_;
        bool affine_ln_; // If true, LN has its own gamma_ln, beta_ln (managed by nn::LayerNorm module)
        torch::nn::LayerNorm layer_norm_{nullptr}; // To be initialized

        // Parameters for learning the switching weights (importance scores)
        // These are typically small linear layers operating on mean channel activations.
        // For simplicity, SN paper uses mean_weight and var_weight for BN, and simple params for IN/LN weights.
        // A common way: fc layers on global channel stats.
        // Let's use a simpler approach for scores: learnable per-channel logits directly.
        // This is similar to how `SparseSwitchableNorm` was implemented.
        // The original SN paper has a more complex scheme for deriving weights involving E[x] and Var[x] of current batch.
        // For simplicity, direct learnable logits per channel for BN, IN, LN.
        torch::Tensor mean_weight_logits_; // Logits for weights based on mean (often for BN)
        torch::Tensor var_weight_logits_; // Logits for weights based on var (often for BN)
        // The paper's weight scheme is:
        // w_bn = softmax(lambda_mean_bn * E[x_ch] + lambda_var_bn * Var[x_ch])
        // w_in = softmax(lambda_in)
        // w_ln = softmax(lambda_ln)
        // This is quite involved.
        // A simpler variant (used in some implementations):
        // Learn 3 scores per channel, then softmax.
        torch::Tensor switching_logits_; // Shape: (num_features, kNumNormalizers)
    };
}
