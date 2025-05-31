#include "include/losses/multi_loss.h"

namespace xt::losses
{
    torch::Tensor multi_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto MultiLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::multi_loss(torch::zeros(10));
    }
}
