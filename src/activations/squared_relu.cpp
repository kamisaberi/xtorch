#include "include/activations/squared_relu.h"

namespace xt::activations
{
    torch::Tensor squared_relu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SquaredReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::squared_relu(torch::zeros(10));
    }
}
