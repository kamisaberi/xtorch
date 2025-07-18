#include <activations/tanh_exp.h>

namespace xt::activations
{
    torch::Tensor tanh_exp(const torch::Tensor& x)
    {
        return x * torch::tanh(torch::exp(x));
    }

    auto TanhExp::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::tanh_exp(torch::zeros(10));
    }
}
