#include "include/losses/piou_loss.h"

namespace xt::losses
{
    torch::Tensor piou_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto PIoULoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::piou_loss(torch::zeros(10));
    }
}
