#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor wgan_gp_loss(torch::Tensor x);
    class WGANGPLoss : xt::Module
    {
    public:
        WGANGPLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
