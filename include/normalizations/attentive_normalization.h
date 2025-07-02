#pragma once

#include "common.h"

namespace xt::norm
{
    struct AttentiveNorm : xt::Module
    {
    public:
        AttentiveNorm(int64_t num_features, double eps = 1e-5, int64_t attention_reduction_ratio = 8);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_;
        double eps_;
        int64_t num_norm_candidates_ = 3; // IN, LN, BN
        int64_t attention_reduction_ratio_;

        // Normalization layers
        torch::nn::InstanceNorm2d instance_norm_{nullptr};
        torch::nn::LayerNorm layer_norm_{nullptr}; // LayerNorm needs normalized_shape
        torch::nn::BatchNorm2d batch_norm_{nullptr};

        // Attention Gate layers
        torch::nn::AdaptiveAvgPool2d avg_pool_{nullptr};
        torch::nn::Conv2d fc1_{nullptr};
        torch::nn::ReLU relu_{nullptr};
        torch::nn::Conv2d fc2_{nullptr};
        torch::nn::Softmax softmax_{nullptr}; // Softmax along the candidates dimension

    };
}
