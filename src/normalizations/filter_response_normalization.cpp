#include "include/normalizations/filter_response_normalization.h"

namespace xt::norm
{
    auto FilterResponseNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
