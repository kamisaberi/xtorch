#include "include/activations/m_arcsinh.h"

namespace xt::activations
{
    torch::Tensor m_arcsinh(const torch::Tensor& x, double m)
    {
        TORCH_CHECK(m > 0, "Parameter m must be positive.");
        return torch::asinh(m * x);
    }

    auto MArcsinh::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::m_arcsinh(torch::zeros(10));
    }
}
