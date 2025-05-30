#include "include/activations/delu.h"

namespace xt::activations
{
    torch::Tensor delu(torch::Tensor x)
    {
        return torch::zeros(10);
    }


    auto DELU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::delu(torch::zeros(10));
    }
}
