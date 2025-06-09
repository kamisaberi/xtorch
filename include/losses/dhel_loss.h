#pragma once
#include "common.h"


namespace xt::losses
{

    torch::Tensor dhel_loss(torch::Tensor x);
    class DHEL : xt::Module
    {
    public:
        DHELLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
