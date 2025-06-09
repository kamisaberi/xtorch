#include "include/losses/dhel_loss.h"

namespace xt::losses
{
    torch::Tensor dhel_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DHELLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dhel(torch::zeros(10));
    }
}
