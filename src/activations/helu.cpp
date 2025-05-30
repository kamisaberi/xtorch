#include "include/activations/helu.h"

namespace xt::activations
{
    torch::Tensor helu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto HeLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::helu(torch::zeros(10));
    }
}
