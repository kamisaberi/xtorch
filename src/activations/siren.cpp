#include "include/activations/siren.h"

namespace xt::activations
{
    torch::Tensor siren(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Siren::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::siren(torch::zeros(10));
    }
}
