#include <activations/nlsig.h>

namespace xt::activations
{
    torch::Tensor nlsig(const torch::Tensor& x, double a, double b)
    {
        torch::Tensor x_abs = torch::abs(x);
        torch::Tensor result = (x / (a + b * x_abs));
        return result;
    }

    auto NLSIG::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::nlsig(torch::zeros(10));
    }
}
