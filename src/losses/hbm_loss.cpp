#include "include/losses/hbm_loss.h"

namespace xt::losses
{
    torch::Tensor hbm_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto HBMLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::hbm_loss(torch::zeros(10));
    }
}
