#include "include/normalizations/layer_scale.h"

namespace xt::norm
{
    auto LayerScale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
