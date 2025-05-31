#include "include/normalizations/cosine_normalization.h"

namespace xt::norm
{
    auto CosineNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
