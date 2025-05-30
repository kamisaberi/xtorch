#include "include/activations/asaf.h"

namespace xt::activations
{
    torch::Tensor asaf(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ASAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::aglu(torch::zeros(10));
    }
}
