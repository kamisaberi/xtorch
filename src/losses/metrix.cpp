#include "include/losses/metrix.h"

namespace xt::losses
{
    torch::Tensor metrix(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Metrix::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::metrix(torch::zeros(10));
    }
}
