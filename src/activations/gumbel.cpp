#include "include/activations/gumbel.h"

namespace xt::activations
{
    torch::Tensor gumbel(const torch::Tensor& x, double beta)
    {
        return x * torch::exp(-torch::exp(-(x / beta)));
    }

    auto Gumbel::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::gumbel(torch::zeros(10));
    }
}
