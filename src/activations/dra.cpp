#include <activations/dra.h>

namespace xt::activations
{
    torch::Tensor dra(const torch::Tensor x, double alpha ) {
        torch::Tensor x_cubed = torch::pow(x, 3.0);
        torch::Tensor arg_sigmoid = alpha * x_cubed;
        torch::Tensor sig_val = torch::sigmoid(arg_sigmoid);
        torch::Tensor result = x * sig_val;
        return result;
    }

    auto DRA::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::dra(torch::zeros(10));
    }
}
