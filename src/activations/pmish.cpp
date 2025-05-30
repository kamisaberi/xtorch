#include "include/activations/pmish.h"

namespace xt::activations
{
    torch::Tensor pmish(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto PMish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::pmish(torch::zeros(10));
    }
}
