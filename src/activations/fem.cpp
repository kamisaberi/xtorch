#include <activations/fem.h>

namespace xt::activations
{
    torch::Tensor fem(const torch::Tensor x, double alpha, double beta)
    {
        torch::Tensor x_plus_alpha = x + alpha;
        torch::Tensor softplus_term = torch::softplus(x_plus_alpha); // softplus(y) = ln(1 + exp(y))
        torch::Tensor tanh_term = torch::tanh(softplus_term);
        torch::Tensor result = x * tanh_term + beta * x;
        return result;
    }

    auto FEM::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::fem(torch::zeros(10));
    }
}
