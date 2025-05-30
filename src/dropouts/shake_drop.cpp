#include "include/dropouts/shake_drop.h"

namespace xt::dropouts
{
    torch::Tensor shake_drop(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ShakeDrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::shake_drop(torch::zeros(10));
    }
}
