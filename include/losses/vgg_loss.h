#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor vgg_loss(torch::Tensor x);
    class VGGLoss : xt::Module
    {
    public:
        VGGLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
