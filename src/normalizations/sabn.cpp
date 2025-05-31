#include "include/normalizations/sabn.h"

namespace xt::norm
{
    auto SABN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
