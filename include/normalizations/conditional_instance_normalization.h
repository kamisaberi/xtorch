#pragma once

#include "common.h"

namespace xt::norm
{
    struct ConditionalInstanceNorm : xt::Module
    {
    public:
        ConditionalInstanceNorm(int64_t num_features,
                                int64_t cond_embedding_dim,
                                int64_t cond_hidden_dim = 0, // 0 means no hidden layer for cond net
                                double eps = 1e-5);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_; // Number of features in input x (channels)
        int64_t cond_embedding_dim_; // Dimensionality of the conditioning input vector
        double eps_;
        int64_t cond_hidden_dim_; // Hidden dimension for the conditioning network
        // Conditioning network layers (to produce gamma and beta)
        torch::nn::Linear fc_cond1_{nullptr}; // Optional first layer
        torch::nn::Linear fc_cond_out_{nullptr}; // Output layer for gamma and beta
    };
}
