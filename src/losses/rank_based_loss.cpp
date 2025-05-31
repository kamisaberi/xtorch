#include "include/losses/rank_based_loss.h"

namespace xt::losses
{
    torch::Tensor rank_based_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto RankBasedLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::rank_based_loss(torch::zeros(10));
    }
}
