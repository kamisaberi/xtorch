#include "include/dropouts/targeted_dropout.h"

namespace xt::dropouts
{
    torch::Tensor targeted_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto TargetedDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::targeted_dropout(torch::zeros(10));
    }
}
