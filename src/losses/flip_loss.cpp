#include "include/losses/flip.h"

namespace xt::losses
{
    torch::Tensor flip(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto FLIP::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::flip(torch::zeros(10));
    }
}
