#include "include/activations/taaf.h"

namespace xt::activations
{
    torch::Tensor taaf(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto TAAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::taaf(torch::zeros(10));
    }
}
