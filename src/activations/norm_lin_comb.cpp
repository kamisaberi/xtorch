#include "include/activations/norm_lin_comb.h"

namespace xt::activations
{
    torch::Tensor norm_lin_comb(
        const torch::Tensor& x,
        const std::vector<std::function<torch::Tensor(const torch::Tensor&)>>& base_functions,
        const torch::Tensor& coefficients, // Shape (num_base_functions)
        double eps
    )
    {
        TORCH_CHECK(!base_functions.empty(), "base_functions vector cannot be empty.");
        TORCH_CHECK(coefficients.dim() == 1, "coefficients must be a 1D tensor.");
        TORCH_CHECK(static_cast<int64_t>(base_functions.size()) == coefficients.size(0),
                    "Number of base functions must match the number of coefficients.");

        torch::Tensor sum_activations = torch::zeros_like(x);
        torch::Tensor sum_sq_coeffs = torch::zeros({1}, x.options()); // Scalar for sum of squares

        for (size_t i = 0; i < base_functions.size(); ++i)
        {
            torch::Tensor activated_x = base_functions[i](x);
            sum_activations += coefficients[i] * activated_x;
            sum_sq_coeffs += coefficients[i] * coefficients[i];
        }

        torch::Tensor norm_factor = torch::sqrt(sum_sq_coeffs + eps);

        return sum_activations / norm_factor;
    }

    auto NormLinComb::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        // return xt::activations::norm_lin_comb(torch::zeros(10));
    }
}
