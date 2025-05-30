//TODO SHOULD IMPLEMENT
#include "include/activations/aglu.h"

namespace xt::activations
{
    torch::Tensor aglu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto AGLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::aglu(torch::zeros(10));
    }
}
