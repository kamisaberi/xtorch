#include "include/dropouts/variational_gaussian_dropout.h"

namespace xt::dropouts
{
    torch::Tensor variational_gaussian_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto VariationalGaussianDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::variational_gaussian_dropout(torch::zeros(10));
    }
}
