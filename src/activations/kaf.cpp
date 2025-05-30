#include "include/activations/kaf.h"

namespace xt::activations
{
    torch::Tensor kaf(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto KAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::kaf(torch::zeros(10));
    }
}
