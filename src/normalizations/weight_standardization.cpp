#include "include/normalizations/weight_standardization.h"

namespace xt::norm
{
    auto WeightStandardization::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
