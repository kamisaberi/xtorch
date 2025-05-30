#include "include/activations/evo_norms.h"

namespace xt::activations
{
    torch::Tensor evo_norms(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto EvoNorms::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::evo_norms(torch::zeros(10));
    }
}
