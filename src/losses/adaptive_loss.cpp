#include "include/losses/adaptive_loss.h"

namespace xt::losses
{
    torch::Tensor adaptive_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto AdaptiveLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::adaptive_loss(torch::zeros(10));
    }
}
