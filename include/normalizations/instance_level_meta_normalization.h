#pragma once

#include "common.h"

namespace xt::norm
{
    struct InstanceLevelMetaNorm : xt::Module
    {
    public:
        InstanceLevelMetaNorm(int64_t num_features,
                              int64_t meta_hidden_dim = 0, // 0 or less means direct map after pooling
                              double eps_in = 1e-5);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_; // Number of features in input x (channels)
        double eps_in_; // Epsilon for Instance Normalization
        int64_t meta_hidden_dim_; // Hidden dimension for the meta-network

        // Meta-network layers (MLP to produce gamma and beta from instance features)
        // It will take pooled features from x as input.
        torch::nn::AdaptiveAvgPool2d avg_pool_{nullptr}; // For NCHW inputs
        torch::nn::Linear fc_meta1_{nullptr};
        torch::nn::Linear fc_meta_out_{nullptr};
    };
}
