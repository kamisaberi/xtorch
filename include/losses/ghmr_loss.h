#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor ghmr_loss(torch::Tensor x);
    class GHMRLoss : xt::Module
    {
    public:
        GHMRLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
