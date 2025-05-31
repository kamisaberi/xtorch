#include "include/losses/generalized_focal_loss.h"

namespace xt::losses
{
    torch::Tensor generalized_focal_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GeneralizedFocalLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::generalized_focal_loss(torch::zeros(10));
    }
}
