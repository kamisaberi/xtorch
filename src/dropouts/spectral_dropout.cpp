#include "include/dropouts/spectral_dropout.h"

namespace xt::dropouts
{
    torch::Tensor spectral_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SpectralDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::spectral_dropout(torch::zeros(10));
    }
}
