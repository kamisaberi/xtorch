#include "include/activations/nfn.h"

namespace xt::activations
{
    torch::Tensor nfn(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto NFN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::nfn(torch::zeros(10));
    }
}
