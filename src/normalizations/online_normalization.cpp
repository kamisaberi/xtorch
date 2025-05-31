#include "include/normalizations/online_normalization.h"

namespace xt::norm
{
    auto OnlineNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
