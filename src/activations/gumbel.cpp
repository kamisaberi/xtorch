#include "include/activations/gumbel.h"

namespace xt::activations
{
    torch::Tensor gumbel(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Gumbel::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::gumbel(torch::zeros(10));
    }
}
