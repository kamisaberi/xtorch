#include <activations/smish.h>

namespace xt::activations
{
    torch::Tensor smish(const torch::Tensor& x, double alpha, double beta)
    {
        return alpha * x * torch::tanh(torch::softplus(beta * x));
    }

    auto Smish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::smish(torch::zeros(10));
    }
}
