#include "include/normalizations/gradient_normalization.h"

namespace xt::norm
{
    auto GradientNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
