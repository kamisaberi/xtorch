#include "include/activations/gcu.h"

namespace xt::activations
{
    torch::Tensor gcu(torch::Tensor x)
    {
        return x * torch::cos(x);
    }

    auto GCU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::gcu(torch::zeros(10));
    }
}
