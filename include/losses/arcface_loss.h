#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor arcface_loss(torch::Tensor x);

    class ArcFaceLoss : xt::Module
    {
    public:
        ArcFaceLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
