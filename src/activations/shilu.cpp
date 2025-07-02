#include "include/activations/shilu.h"

namespace xt::activations
{
    torch::Tensor shilu(const torch::Tensor& x, double a, double b)
    {
        torch::Tensor x_plus_b = x + b;
        torch::Tensor result = x_plus_b * torch::sigmoid(a * x_plus_b);
        return result;
    }

    auto ShiLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::shilu(torch::zeros(10));
    }
}
