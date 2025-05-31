#include "include/losses/varifocal_loss.h"

namespace xt::losses
{
    torch::Tensor varifocal_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto VarifocalLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::varifocal_loss(torch::zeros(10));
    }
}
