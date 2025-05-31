#include "include/normalizations/switchable_normalization.h"

namespace xt::norm
{
    auto SwitchableNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
