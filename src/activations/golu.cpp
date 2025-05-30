#include "include/activations/golu.h"

namespace xt::activations
{
    torch::Tensor golu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GoLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::golu(torch::zeros(10));
    }
}
