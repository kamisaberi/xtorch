#include "include/activations/coslu.h"

namespace xt::activations
{
    torch::Tensor coslu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto CosLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::coslu(torch::zeros(10));
    }
}
