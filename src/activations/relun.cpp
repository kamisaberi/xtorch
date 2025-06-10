#include "include/activations/relun.h"

namespace xt::activations
{
    torch::Tensor relun(const torch::Tensor& x, double n_val)
    {
        TORCH_CHECK(n_val > 0, "n_val (upper bound) must be positive.");
        return torch::clamp(x, 0.0, n_val);
    }

    auto ReLUN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::relun(torch::zeros(10));
    }
}
