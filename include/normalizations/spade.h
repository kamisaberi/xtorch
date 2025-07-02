#pragma once

#include "common.h"

namespace xt::norm
{
    struct SPADE : xt::Module
    {
    public:
        SPADE(int64_t norm_num_features,
              int64_t seg_map_channels,
              int64_t hidden_channels_mlp = 128, // Common choice
              double eps_bn = 1e-5,
              double momentum_bn = 0.1);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t norm_num_features_; // Number of features in input x (channels to be normalized)
        int64_t seg_map_channels_; // Number of channels in the input segmentation map
        double eps_bn_;
        double momentum_bn_;

        // Batch Normalization components (without learnable affine parameters)
        torch::nn::BatchNorm2d batch_norm_{nullptr}; // Will be configured with affine=false

        // CNN for processing the segmentation map to produce gamma and beta
        // Usually a few conv layers. For simplicity, let's use two.
        // The number of hidden channels in this MLP can be a hyperparameter.
        int64_t hidden_channels_mlp_;
        torch::nn::Conv2d mlp_shared_conv1_{nullptr};
        // Separate conv layers for predicting gamma and beta
        torch::nn::Conv2d mlp_gamma_conv2_{nullptr};
        torch::nn::Conv2d mlp_beta_conv2_{nullptr};
    };
}
