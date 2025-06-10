#include "include/activations/colu.h"

namespace xt::activations
{
    torch::Tensor colu(const torch::Tensor& x, double M_val)
    {
        torch::Tensor result = torch::where(
            x < -M_val,
            2.0 * x + 2.0 * M_val,
            x + M_val
        );
        return result;
    }

    auto CoLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::colu(torch::zeros(10));
    }
}
