#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor early_exiting_loss(torch::Tensor x);
    class EarlyExitingLoss : xt::Module
    {
    public:
        EarlyExitingLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
