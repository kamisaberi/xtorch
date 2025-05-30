#include "include/activations/smooth_step.h"

namespace xt::activations
{
    torch::Tensor smooth_step(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SmoothStep::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::smooth_step(torch::zeros(10));
    }
}
