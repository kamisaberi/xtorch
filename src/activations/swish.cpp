#include "include/activations/swish.h"

namespace xt::activations
{
    torch::Tensor swish(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Swish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::swish(torch::zeros(10));
    }
}
