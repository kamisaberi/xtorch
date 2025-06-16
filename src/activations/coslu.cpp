#include "include/activations/coslu.h"

namespace xt::activations
{
    torch::Tensor coslu(torch::Tensor&x)
    {
        torch::Tensor positive_part = x * torch::cos(x);
        torch::Tensor negative_part = x * torch::exp(x);
        torch::Tensor result = torch::where(x >= 0, positive_part, negative_part);
        return result;
    }

    auto CosLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::coslu(torch::zeros(10));
    }
}
