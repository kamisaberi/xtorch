#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor dice_loss(torch::Tensor x);
    class DiceLoss : xt::Module
    {
    public:

        DiceLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
