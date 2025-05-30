#include "include/activations/srelu.h"

namespace xt::activations
{
    torch::Tensor srelu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::srelu(torch::zeros(10));
    }
}
