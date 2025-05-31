#include "include/normalizations/weight_demodulation.h"

namespace xt::norm
{
    auto WeightDemodulization::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
