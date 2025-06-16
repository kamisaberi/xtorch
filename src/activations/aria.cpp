#include "include/activations/aria.h"

namespace xt::activations
{
    torch::Tensor aria(const torch::Tensor x, double alpha, double beta)
    {
        torch::Tensor x_pow_alpha = torch::pow(x, alpha);
        torch::Tensor arg_tanh = beta * x_pow_alpha;
        torch::Tensor tanh_val = torch::tanh(arg_tanh);
        torch::Tensor gate = (1.0 + tanh_val) / 2.0;
        return x * gate;
    }

    auto ARiA::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::aria(torch::zeros(10));
    }
}
