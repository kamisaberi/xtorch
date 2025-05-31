#include "include/normalizations/local_contrast_normalization.h"

namespace xt::norm
{
    auto LocalContrastNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
