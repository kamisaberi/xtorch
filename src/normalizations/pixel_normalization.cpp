#include "include/normalizations/pixel_normalization.h"

namespace xt::norm
{
    auto PixelNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
