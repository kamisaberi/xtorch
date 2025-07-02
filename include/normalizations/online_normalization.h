#pragma once

#include "common.h"

namespace xt::norm
{
    struct OnlineNorm : xt::Module
    {
    public:
        OnlineNorm(int64_t num_features,
                   double eps = 1e-5,
                   double momentum_mu = 0.1, // Paper suggests 0.1 or 0.01 for mu
                   double momentum_sigma = 0.1, // Paper suggests 0.1 or 0.001 for sigma
                   bool affine_gn = true);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_;
        double eps_;
        double momentum_mu_; // Momentum for online mean update
        double momentum_sigma_; // Momentum for online variance update
        bool affine_gn_; // Whether the g(.) affirmative part has learnable alpha, beta

        // Learnable parameters for the affirmative part g(x) = alpha * x + beta
        torch::Tensor alpha_; // Scale for g(x)
        torch::Tensor beta_; // Shift for g(x)

        // Buffers for online statistics (for h(y) part)
        // These are E[g(x)] and Var[g(x)]
        torch::Tensor online_mu_;
        torch::Tensor online_sigma_sq_; // online variance

        // For controlling the update during the very first batch
        torch::Tensor num_batches_tracked_; // Similar to BN's, but for initializing online stats
    };
}
