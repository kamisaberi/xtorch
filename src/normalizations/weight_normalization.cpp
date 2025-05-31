#include "include/normalizations/weight_normalization.h"

namespace xt::norm
{
    auto WeightNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
