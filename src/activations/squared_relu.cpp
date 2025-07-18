#include <activations/squared_relu.h>

namespace xt::activations
{
    torch::Tensor squared_relu(const torch::Tensor& x)
    {
        torch::Tensor relu_x = torch::relu(x);
        return relu_x * relu_x;
    }

    auto SquaredReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::squared_relu(torch::zeros(10));
    }
}
