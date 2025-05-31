#include "include/normalizations/mixture_normalization.h"

namespace xt::norm
{
    auto MixtureNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
