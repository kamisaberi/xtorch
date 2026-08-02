#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor wgan_gp_loss(torch::Tensor x);
    class WGANGPLoss : xt::Module
    {
    public:


    private:
    };
}
