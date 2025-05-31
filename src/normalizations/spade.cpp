#include "include/normalizations/spade.h"

namespace xt::norm
{
    auto SPADE::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
