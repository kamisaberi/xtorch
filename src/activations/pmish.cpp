#include "include/activations/pmish.h"

namespace xt::activations
{
    torch::Tensor pmish(const torch::Tensor& x, double alpha , double beta )
    {
        return (x / alpha) * torch::tanh(torch::softplus(alpha * x + beta));
    }

    auto PMish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::pmish(torch::zeros(10));
    }
}
