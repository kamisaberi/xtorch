#include "include/activations/phish.h"

namespace xt::activations
{
    torch::Tensor phish(const torch::Tensor& x, double a, double b )
    {
        return x * torch::tanh(a * x) + b * x * (1.0 - torch::sigmoid(x));
    }

    auto Phish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::phish(torch::zeros(10));
    }
}
