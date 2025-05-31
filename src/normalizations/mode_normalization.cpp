#include "include/normalizations/mode_normalization.h"

namespace xt::norm
{
    auto ModeNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
