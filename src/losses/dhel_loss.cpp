#include "include/losses/dhel.h"

namespace xt::losses
{
    torch::Tensor dhel(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DHEL::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dhel(torch::zeros(10));
    }
}
