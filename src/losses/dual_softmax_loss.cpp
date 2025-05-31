#include "include/losses/dual_softmax_loss.h"

namespace xt::losses
{
    torch::Tensor dual_softmax_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DualSoftmaxLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dual_softmax_loss(torch::zeros(10));
    }
}
