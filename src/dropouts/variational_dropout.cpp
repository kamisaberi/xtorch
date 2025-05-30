#include "include/dropouts/variational_dropout.h"

namespace xt::dropouts
{
    torch::Tensor variational_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto VariationalDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::variational_dropout(torch::zeros(10));
    }
}
