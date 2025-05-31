#include "include/normalizations/self_norm.h"

namespace xt::norm
{
    auto SelfNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
