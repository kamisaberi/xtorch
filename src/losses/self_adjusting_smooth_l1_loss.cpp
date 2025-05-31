#include "include/losses/self_adjusting_smooth_l1_loss.h"

namespace xt::losses
{
    torch::Tensor self_adjusting_smooth_l1_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SelfAdjustingSmoothL1Loss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::self_adjusting_smooth_l1_loss(torch::zeros(10));
    }
}
