#include "include/normalizations/spectral_normalization.h"

namespace xt::norm
{
    auto SpectralNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
