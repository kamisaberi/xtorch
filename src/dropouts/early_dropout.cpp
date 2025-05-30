#include "include/dropouts/early_dropout.h"

namespace xt::dropouts
{
    torch::Tensor early_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto EarlyDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::early_dropout(torch::zeros(10));
    }
}
