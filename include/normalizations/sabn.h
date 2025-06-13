#pragma once

#include "common.h"

namespace xt::norm
{
    struct SABN : xt::Module
    {
    public:
        SABN(int64_t num_features,
             double eps = 1e-5,
             double momentum = 0.1,
             double leaky_relu_slope = 0.01);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_;
        double eps_;
        double momentum_;
        // Affine parameters are part of each BN branch

        // Activation parameters (for LeakyReLU)
        double leaky_relu_slope_;

        // Components for two Batch Normalization branches (BN1, BN2)
        // Branch 1
        torch::Tensor running_mean1_;
        torch::Tensor running_var1_;
        torch::Tensor gamma1_;
        torch::Tensor beta1_;
        torch::Tensor num_batches_tracked1_;

        // Branch 2
        torch::Tensor running_mean2_;
        torch::Tensor running_var2_;
        torch::Tensor gamma2_;
        torch::Tensor beta2_;
        torch::Tensor num_batches_tracked2_;

        // Learnable mixing weights (for combining outputs of BN1 and BN2)
        // We'll learn `mixing_logits_` and pass them through softmax to get weights for each branch.
        // For two branches, we only need one set of logits per channel, representing weight for branch1.
        // Weight for branch2 will be (1 - weight_branch1).
        // Or, more generally, K logits for K branches, then softmax. Let's use 2 logits for 2 branches.
        torch::Tensor mixing_logits_; // Shape (1, C, 1, 1, num_branches=2)
    };
}
