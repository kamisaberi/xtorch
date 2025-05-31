#include "include/losses/dice_loss.h"

namespace xt::losses
{
    torch::Tensor dice_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DiceLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dice_loss(torch::zeros(10));
    }
}
