#include "include/activations/lin_comb.h"

namespace xt::activations
{
    torch::Tensor lin_comb(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto LinComb::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::lin_comb(torch::zeros(10));
    }
}
