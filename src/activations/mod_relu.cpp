#include "include/activations/mod_relu.h"

namespace xt::activations
{
    torch::Tensor mod_relu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ModReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::mod_relu(torch::zeros(10));
    }
}
