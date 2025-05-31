#include "include/losses/oa_loss.h"

namespace xt::losses
{
    torch::Tensor oa_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto OALoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::oa_loss(torch::zeros(10));
    }
}
