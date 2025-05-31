#include "include/normalizations/cross_norm.h"

namespace xt::norm
{
    auto CrossNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
