#include "include/dropouts/checkerboard_dropout.h"

namespace xt::dropouts
{
    torch::Tensor checkerboard_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto CheckerboardDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::checkerboard_dropout(torch::zeros(10));
    }
}
