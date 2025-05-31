#include "include/losses/focal_loss.h"

namespace xt::losses
{
    torch::Tensor focal_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto FocalLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::focal_loss(torch::zeros(10));
    }
}
