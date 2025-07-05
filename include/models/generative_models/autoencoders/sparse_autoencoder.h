#pragma once

#include "../../common.h"


namespace xt::models
{
    struct SparseAutoEncoder : xt::Cloneable<SparseAutoEncoder>
    {
    private:

    public:
        SparseAutoEncoder(int latent_dim = 128);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;

    private:
        torch::nn::Conv2d enc_conv1{nullptr}, enc_conv2{nullptr}, enc_conv3{nullptr};
        torch::nn::Linear enc_fc{nullptr};
        torch::nn::Linear dec_fc{nullptr};
        torch::nn::ConvTranspose2d dec_conv1{nullptr}, dec_conv2{nullptr}, dec_conv3{nullptr};
        torch::nn::ReLU relu{nullptr};
        torch::nn::Sigmoid sigmoid{nullptr};
    };
}
