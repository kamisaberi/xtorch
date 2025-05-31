#include "include/normalizations/sparse_switchable_normalization.h"

namespace xt::norm
{
    auto SparseSwitchableNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
