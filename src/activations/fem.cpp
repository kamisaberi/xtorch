#include "include/activations/fem.h"

namespace xt::activations
{
    torch::Tensor fem(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto FEM::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::fem(torch::zeros(10));
    }
}
