#include "include/dropouts/gaussian_dropout.h"

namespace xt::dropouts
{
    torch::Tensor gaussian_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GaussianDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::gaussian_dropout(torch::zeros(10));
    }
}
