#include <activations/mish.h>

namespace xt::activations
{
    torch::Tensor mish(torch::Tensor x)
    {
        return x * torch::tanh(torch::softplus(x));
    }

    auto Mish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::mish(torch::zeros(10));
    }
}
