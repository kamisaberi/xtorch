#pragma once

#include "common.h"

namespace xt::norm
{
    struct CINCFlow : xt::Module
    {
    public:
        CINCFlow(int64_t num_features, int64_t cond_embedding_dim, int64_t hidden_dim_cond_net = 0);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_;          // Number of features in the input x (e.g., channels)
        int64_t cond_embedding_dim_;    // Dimensionality of the conditioning input vector
        int64_t hidden_dim_cond_net_;   // Hidden dimension for the conditioning network

        // Conditioning network layers (MLP to produce scale and bias)
        torch::nn::Linear fc1_cond_{nullptr};
        torch::nn::Linear fc2_cond_{nullptr};
        // No explicit activation after fc1_cond to allow more direct mapping,
        // or you could add one like ReLU.
        // The output of fc2_cond will be split into log_scale and bias.

    };
}
