#include "include/activations/poly.h"

namespace xt::activations
{
    torch::Tensor poly(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Poly::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::poly(torch::zeros(10));
    }
}
