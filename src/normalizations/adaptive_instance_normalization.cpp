#include "include/normalizations/adaptive_instance_normalization.h"

namespace xt::norm
{
    auto AdaptiveInstanceNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
