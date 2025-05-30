#include "include/activations/hard_swish.h"

namespace xt::activations
{
    torch::Tensor hard_swich(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto HardSwish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::hard_swich(torch::zeros(10));
    }
}
