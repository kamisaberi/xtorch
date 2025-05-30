#include "include/dropouts/spatial_dropout.h"

namespace xt::dropouts
{
    torch::Tensor spatial_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SpatialDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::spatial_dropout(torch::zeros(10));
    }
}
