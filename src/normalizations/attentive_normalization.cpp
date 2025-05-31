#include "include/normalizations/attentive_normalization.h"

namespace xt::norm
{
    auto AttentiveNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
