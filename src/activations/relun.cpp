#include "include/activations/relun.h"

namespace xt::activations
{
    torch::Tensor relun(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ReLUN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::relun(torch::zeros(10));
    }
}
