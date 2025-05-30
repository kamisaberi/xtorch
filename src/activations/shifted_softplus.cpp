#include "include/activations/shifted_softplus.h"

namespace xt::activations
{
    torch::Tensor shifted_softplus(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ShiftedSoftplus::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::shifted_softplus(torch::zeros(10));
    }
}
