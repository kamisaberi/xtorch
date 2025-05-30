#include "include/activations/geglu.h"

namespace xt::activations
{
    torch::Tensor geglu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GeGLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::geglu(torch::zeros(10));
    }
}
