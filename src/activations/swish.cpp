#include <activations/swish.h>

namespace xt::activations
{
    torch::Tensor swish(const torch::Tensor& x, double beta )
    {
        return x * torch::sigmoid(beta * x);
    }

    auto Swish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::swish(torch::zeros(10));
    }
}
