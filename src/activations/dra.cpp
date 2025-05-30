#include "include/activations/dra.h"

namespace xt::activations
{
    torch::Tensor dra(torch::Tensor x)
    {
        return torch::zeros(10);
    }


    auto DRA::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::dra(torch::zeros(10));
    }
}
