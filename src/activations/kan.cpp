#include "include/activations/kan.h"

namespace xt::activations
{
    torch::Tensor kan(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto KAN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::kan(torch::zeros(10));
    }
}
