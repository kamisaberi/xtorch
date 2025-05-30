#include "include/activations/nipuna.h"

namespace xt::activations
{
    torch::Tensor nipuna(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Nipuna::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::nipuna(torch::zeros(10));
    }
}
