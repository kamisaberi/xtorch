#include "include/losses/dynamic_smooth_l1_loss.h"

namespace xt::losses
{
    torch::Tensor dynamic_smooth_l1_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DynamicSmoothL1Loss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dynamic_smooth_l1_loss(torch::zeros(10));
    }
}
