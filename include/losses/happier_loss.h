#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor happier_loss(torch::Tensor x);
    class HAPPIERLoss : xt::Module
    {
    public:
        HAPPIERLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
