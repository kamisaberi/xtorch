#include "include/activations/mod_relu.h"

namespace xt::activations
{
    torch::Tensor mod_relu(const torch::Tensor& x, const torch::Tensor& b)
    {
        TORCH_CHECK(x.sizes() == b.sizes() || b.numel() == 1,
                    "Input x and bias b must have the same shape, or b must be a scalar.");

        torch::Tensor abs_x_plus_b = torch::abs(x) + b;
        torch::Tensor result = torch::relu(x) * (abs_x_plus_b / (torch::abs(abs_x_plus_b) + 1e-7));
        return result;
    }

    auto ModReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::mod_relu(torch::zeros(10), torch::zeros(10));
    }
}
