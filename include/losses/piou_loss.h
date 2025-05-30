#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor piou_loss(torch::Tensor x);
    class PIoULoss : xt::Module
    {
    public:
        PIoULoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
