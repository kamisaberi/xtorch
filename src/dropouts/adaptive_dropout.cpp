#include "include/dropouts/adaptive_dropout.h"

namespace xt::dropouts
{
    torch::Tensor adaptive_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto AdaptiveDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::adaptive_dropout(torch::zeros(10));
    }
}
