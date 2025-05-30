#include "include/activations/scaled_soft_sign.h"

namespace xt::activations
{
    torch::Tensor scaled_soft_sign(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ScaledSoftSign::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::scaled_soft_sign(torch::zeros(10));
    }
}
