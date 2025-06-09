#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor dice_bce_loss(torch::Tensor x);
    class DiceBCELoss : xt::Module
    {
    public:

        DiceBCELoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
