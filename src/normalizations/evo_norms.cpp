#include "include/normalizations/evo_norms.h"

namespace xt::norm
{
    auto EvoNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
