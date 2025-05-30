#include "include/activations/colu.h"

namespace xt::activations
{
    torch::Tensor colu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto CoLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::colu(torch::zeros(10));
    }
}
