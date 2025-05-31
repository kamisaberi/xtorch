#include "include/normalizations/local_response_normalization.h"

namespace xt::norm
{
    auto LocalResponseNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
