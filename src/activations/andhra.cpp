//TODO SHOULD IMPLEMENT
#include "include/activations/andhra.h"

namespace xt::activations
{
    torch::Tensor andhra(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ANDHRA::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::andhra(torch::zeros(10));
    }
}
