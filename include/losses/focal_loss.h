#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor focal_loss(torch::Tensor x);
    class FocalLoss : xt::Module
    {
    public:
        FocalLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
