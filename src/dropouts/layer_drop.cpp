#include "include/dropouts/layer_drop.h"

namespace xt::dropouts
{
    torch::Tensor layer_drop(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto LayerDrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::layer_drop(torch::zeros(10));
    }
}
