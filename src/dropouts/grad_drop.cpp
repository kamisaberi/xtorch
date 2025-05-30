#include "include/dropouts/grad_drop.h"

namespace xt::dropouts
{
    torch::Tensor grad_drop(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GradDrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::grad_drop(torch::zeros(10));
    }
}
