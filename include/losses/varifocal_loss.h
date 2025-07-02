#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor varifocal_loss(torch::Tensor x);
    class VarifocalLoss : xt::Module
    {
    public:
        VarifocalLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
