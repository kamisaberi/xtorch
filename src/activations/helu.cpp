#include "include/activations/helu.h"

namespace xt::activations
{
    torch::Tensor helu(const torch::Tensor& x, double alpha, double lambda_param)
    {
        torch::Tensor positive_part = x;
        torch::Tensor negative_part = alpha * (torch::exp(x / lambda_param) - 1.0);

        torch::Tensor result = torch::where(x >= 0, positive_part, negative_part);
        return result;
    }

    auto HeLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::helu(torch::zeros(10));
    }
}
