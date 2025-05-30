#include "include/activations/nlsig.h"

namespace xt::activations
{
    torch::Tensor nlsig(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto NLSIG::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::nlsig(torch::zeros(10));
    }
}
