#include "include/activations/nail_or.h"

namespace xt::activations
{
    torch::Tensor nail_or(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto NailOr::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::nail_or(torch::zeros(10));
    }
}
