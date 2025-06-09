#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor ghmc_loss(torch::Tensor x);
    class GHMCLoss : xt::Module
    {
    public:
        GHMCLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
