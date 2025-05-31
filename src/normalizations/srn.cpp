#include "include/normalizations/srn.h"

namespace xt::norm
{
    auto SRN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
