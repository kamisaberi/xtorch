#include "include/activations/margin_relu.h"

namespace xt::activations
{
    torch::Tensor margin_relu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto MarginReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::margin_relu(torch::zeros(10));
    }
}
