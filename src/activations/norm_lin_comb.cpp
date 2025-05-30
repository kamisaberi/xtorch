#include "include/activations/norm_lin_comb.h"

namespace xt::activations
{
    torch::Tensor norm_lin_comb(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto NormLinComb::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::norm_lin_comb(torch::zeros(10));
    }
}
