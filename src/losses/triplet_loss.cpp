#include "include/losses/triplet_loss.h"

namespace xt::losses
{
    torch::Tensor triplet_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto TripletLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::triplet_loss(torch::zeros(10));
    }
}
