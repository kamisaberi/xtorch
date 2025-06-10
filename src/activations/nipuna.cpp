#include "include/activations/nipuna.h"

namespace xt::activations
{
    torch::Tensor nipuna(const torch::Tensor& x, double a, double b)
    {
        torch::Tensor x_plus_a = x + a;
        torch::Tensor softplus_term = torch::softplus(x_plus_a); // softplus(y) = ln(1 + exp(y))
        torch::Tensor tanh_term = torch::tanh(softplus_term);
        torch::Tensor result = x * tanh_term + b * x * (1.0 - tanh_term);
        return result;
    }

    auto Nipuna::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::nipuna(torch::zeros(10));
    }
}
