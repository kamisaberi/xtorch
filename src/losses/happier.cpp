#include "include/losses/happier.h"

namespace xt::losses
{
    torch::Tensor happier(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto HAPPIER::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::happier(torch::zeros(10));
    }
}
