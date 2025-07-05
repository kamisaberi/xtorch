#pragma once

#include "../../common.h"



namespace xt::models
{
    struct AE : xt::Cloneable<AE>
    {
    private:

    public:
        AE(int latent_dim = 32);

        AE(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;

    private:
        torch::nn::Conv2d enc_conv1{nullptr}, enc_conv2{nullptr};
        torch::nn::Linear enc_fc{nullptr};
        torch::nn::Linear dec_fc{nullptr};
        torch::nn::ConvTranspose2d dec_conv1{nullptr}, dec_conv2{nullptr};
        torch::nn::ReLU relu{nullptr};
        torch::nn::Sigmoid sigmoid{nullptr};

    };

}