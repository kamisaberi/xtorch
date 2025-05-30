#include "include/activations/shilu.h"

namespace xt::activations
{
    torch::Tensor shilu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ShiLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::shilu(torch::zeros(10));
    }
}
