#include "include/normalizations/mpn.h"

namespace xt::norm
{
    auto MPN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
