#include "include/activations/phish.h"

namespace xt::activations
{
    torch::Tensor phish(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Phish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::phish(torch::zeros(10));
    }
}
