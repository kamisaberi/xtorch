#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor dynamic_smooth_l1_loss(torch::Tensor x);
    class DynamicSmoothL1Loss : xt::Module
    {
    public:
        DynamicSmoothL1Loss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
