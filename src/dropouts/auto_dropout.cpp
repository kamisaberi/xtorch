#include "include/dropouts/auto_dropout.h"

namespace xt::dropouts
{
    torch::Tensor auto_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto AutoDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::auto_dropout(torch::zeros(10));
    }
}
