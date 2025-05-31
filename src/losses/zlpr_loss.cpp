#include "include/losses/zlpr_loss.h"

namespace xt::losses
{
    torch::Tensor zlpr_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ZLPRLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::zlpr_loss(torch::zeros(10));
    }
}
