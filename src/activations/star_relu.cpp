#include "include/activations/star_relu.h"

namespace xt::activations
{
    torch::Tensor star_relu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto StarReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::star_relu(torch::zeros(10));
    }
}
