#include "include/normalizations/conditional_instance_normalization.h"

namespace xt::norm
{
    auto ConditionalInstanceNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
