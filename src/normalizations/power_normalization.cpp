#include "include/normalizations/power_normalization.h"

namespace xt::norm
{
    auto PowerNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
