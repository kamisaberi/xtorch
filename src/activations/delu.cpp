#include <activations/delu.h>

namespace xt::activations
{
    torch::Tensor delu(const torch::Tensor x, double alpha, double gamma)
    {
        torch::Tensor elu_component = torch::elu(x, alpha, 1.0, 0.0);
        torch::Tensor result = elu_component + gamma;
        return result;
    }


    auto DELU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::delu(torch::zeros(10));
    }
}
