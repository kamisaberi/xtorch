#include "include/activations/e_swich.h"

namespace xt::activations
{
    torch::Tensor e_swish(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ESwish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::e_swish(torch::zeros(10));
    }
}
