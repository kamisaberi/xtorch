#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor gan_hinge_loss(torch::Tensor x);
    class GANHingeLoss : xt::Module
    {
    public:
        GANHingeLoss() = default;


    private:
    };
}
