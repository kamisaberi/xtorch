#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor gan_hinge_loss(torch::Tensor x);
    class GANHingeLoss : xt::Module
    {
    public:
        GANHingeLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
