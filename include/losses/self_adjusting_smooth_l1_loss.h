#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor self_adjusting_smooth_l1_loss(torch::Tensor x);
    class SelfAdjustingSmoothL1Loss : xt::Module
    {
    public:
        SelfAdjustingSmoothL1Loss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
