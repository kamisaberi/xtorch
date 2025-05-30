#include "include/activations/pau.h"

namespace xt::activations
{
    torch::Tensor pau(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto PAU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::pau(torch::zeros(10));
    }
}
