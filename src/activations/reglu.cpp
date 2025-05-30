#include "include/activations/reglu.h"

namespace xt::activations
{
    torch::Tensor reglu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ReGLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::reglu(torch::zeros(10));
    }
}
