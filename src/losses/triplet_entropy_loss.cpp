#include "include/losses/triplet_entropy_loss.h"

namespace xt::losses
{
    torch::Tensor triplet_entropy_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto TripletEntropyLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::triplet_entropy_loss(torch::zeros(10));
    }
}
