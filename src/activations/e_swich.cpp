#include <activations/e_swich.h>

namespace xt::activations
{
    torch::Tensor e_swish(const torch::Tensor x, double beta) {
        torch::Tensor silu_x = x * torch::sigmoid(x);
        torch::Tensor result = beta * silu_x;
        return result;
    }

    auto ESwish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::e_swish(torch::zeros(10));
    }
}
