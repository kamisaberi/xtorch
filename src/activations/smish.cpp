#include "include/activations/smish.h"

namespace xt::activations
{
    torch::Tensor smish(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Smish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::smish(torch::zeros(10));
    }
}
