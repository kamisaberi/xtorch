#include "include/activations/tanh_exp.h"

namespace xt::activations
{
    torch::Tensor tanh_exp(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto TanhExp::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::tanh_exp(torch::zeros(10));
    }
}
