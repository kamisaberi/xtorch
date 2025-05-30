#include "include/dropouts/fraternal_dropout.h"

namespace xt::dropouts
{
    torch::Tensor fraternal_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto FraternalDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::fraternal_dropout(torch::zeros(10));
    }
}
