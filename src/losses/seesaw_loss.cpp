#include "include/losses/seesaw_loss.h"

namespace xt::losses
{
    torch::Tensor seesaw_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SeesawLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::seesaw_loss(torch::zeros(10));
    }
}
