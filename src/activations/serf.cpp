#include "include/activations/serf.h"

namespace xt::activations
{
    torch::Tensor serf(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Serf::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::serf(torch::zeros(10));
    }
}
