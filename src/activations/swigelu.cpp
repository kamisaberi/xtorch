#include "include/activations/swigelu.h"

namespace xt::activations
{
    torch::Tensor swiglu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SwiGLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::swiglu(torch::zeros(10));
    }
}
