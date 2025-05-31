#include "include/normalizations/activation_normalization.h"

namespace xt::norm
{
    auto ActiveNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
