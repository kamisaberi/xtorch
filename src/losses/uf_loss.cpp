#include "include/losses/uf_loss.h"

namespace xt::losses
{
    torch::Tensor uf_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto UFLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::uf_loss(torch::zeros(10));
    }
}
