#include "include/dropouts/concrete_dropout.h"

namespace xt::dropouts
{
    torch::Tensor concrete_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ConcreteDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::concrete_dropout(torch::zeros(10));
    }
}
