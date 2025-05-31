#include "include/normalizations/clnc_flow.h"

namespace xt::norm
{
    auto CLNCFlow::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
