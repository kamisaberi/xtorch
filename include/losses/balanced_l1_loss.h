#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor balanced_l1_loss(torch::Tensor x);
    class BalancedL1Loss : xt::Module
    {
    public:
        BalancedL1Loss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
    private:
    };
}
