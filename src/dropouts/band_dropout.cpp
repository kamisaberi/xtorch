#include "include/dropouts/band_dropout.h"

namespace xt::dropouts
{
    torch::Tensor band_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto BandDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::band_dropout(torch::zeros(10));
    }
}
