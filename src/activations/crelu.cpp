#include "include/activations/crelu.h"

namespace xt::activations
{
    torch::Tensor crelu(torch::Tensor x)
    {
        return torch::zeros(10);
    }


    auto CReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::crelu(torch::zeros(10));
    }
}
