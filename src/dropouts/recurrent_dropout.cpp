#include "include/dropouts/recurrent_dropout.h"

namespace xt::dropouts
{
    torch::Tensor recurrent_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto RecurrentDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::recurrent_dropout(torch::zeros(10));
    }
}
