#include "include/losses/cycle_consistency_loss.h"

namespace xt::losses
{
    torch::Tensor cycle_consistency_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto CycleConsistencyLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::cycle_consistency_loss(torch::zeros(10));
    }
}
