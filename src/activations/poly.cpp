#include "include/activations/poly.h"

namespace xt::activations
{
    torch::Tensor poly(
        const torch::Tensor& x,
        const torch::Tensor& coefficients // Shape [degree, degree-1, ..., 1, 0]
    )
    {
        TORCH_CHECK(coefficients.dim() == 1, "coefficients must be a 1D tensor.");
        TORCH_CHECK(coefficients.size(0) > 0, "coefficients tensor cannot be empty.");

        int64_t degree = coefficients.size(0) - 1;
        torch::Tensor result = torch::zeros_like(x);

        for (int64_t i = 0; i <= degree; ++i)
        {
            result += coefficients[degree - i] * torch::pow(x, static_cast<double>(i));
        }

        return result;
    }

    auto Poly::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::poly(torch::zeros(10), torch::zeros(10));
    }
}
