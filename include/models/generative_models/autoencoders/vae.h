#pragma once

#include "../../common.h"


namespace xt::models
{
    struct VAE : xt::Cloneable<VAE>
    {
    private:

    public:
        VAE(int latent_dim = 32);


        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    private:
        int latent_dim_;
        torch::nn::Conv2d enc_conv1{nullptr}, enc_conv2{nullptr};
        torch::nn::Linear enc_fc_mu{nullptr}, enc_fc_logvar{nullptr};
        torch::nn::Linear dec_fc{nullptr};
        torch::nn::ConvTranspose2d dec_conv1{nullptr}, dec_conv2{nullptr};
        torch::nn::ReLU relu{nullptr};
        torch::nn::Sigmoid sigmoid{nullptr};

    };

}