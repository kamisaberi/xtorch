#include "include/activations/hermite.h"

namespace xt::activations
{
    torch::Tensor hermite(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Hermite::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::hermite(torch::zeros(10));
    }
}
