#include "include/losses/balanced_l1_loss.h"

namespace xt::losses
{
    torch::Tensor balanced_l1_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto BalancedL1Loss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::balanced_l1_loss(torch::zeros(10));
    }
}
