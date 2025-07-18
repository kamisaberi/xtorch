#include <activations/serlu.h>

namespace xt::activations
{
    torch::Tensor serlu(const torch::Tensor& x, double lambda_serlu, double alpha_serlu)
    {
        torch::Tensor positive_part = lambda_serlu * x;
        torch::Tensor negative_part = lambda_serlu * alpha_serlu * (torch::exp(x) - 1.0);

        torch::Tensor result = torch::where(x > 0, positive_part, negative_part);
        return result;
    }

    auto SERLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::serlu(torch::zeros(10));
    }
}
