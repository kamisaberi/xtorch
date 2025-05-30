#include "include/activations/hard_elish.h"

namespace xt::activations
{
    torch::Tensor hard_elish(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto HardELiSH::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::hard_elish(torch::zeros(10));
    }
}
