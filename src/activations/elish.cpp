#include "include/activations/elish.h"

namespace xt::activations
{
    torch::Tensor elish(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ELiSH::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::elish(torch::zeros(10));
    }
}
