#include "include/losses/flip_loss.h"

namespace xt::losses
{
    torch::Tensor flip_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto FLIPLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::flip_loss(torch::zeros(10));
    }
}
