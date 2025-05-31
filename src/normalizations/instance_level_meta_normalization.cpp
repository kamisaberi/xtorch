#include "include/normalizations/instance_level_meta_normalization.h"

namespace xt::norm
{
    auto InstanceLevelMetaNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
