#include "include/activations/m_arcsinh.h"

namespace xt::activations
{
    torch::Tensor m_arcsinh(torch::Tensor x)
    {
        return torch::zeros(10);
    }


    auto MArcsinh::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::m_arcsinh(torch::zeros(10));
    }
}
