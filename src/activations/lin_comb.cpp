#include "include/activations/lin_comb.h"

namespace xt::activations
{
    torch::Tensor lin_comb(
        const torch::Tensor& x,
        const std::vector<std::function<torch::Tensor(const torch::Tensor&)>>& base_functions,
        const torch::Tensor& coefficients // Shape (num_base_functions)
    )
    {
        TORCH_CHECK(!base_functions.empty(), "base_functions vector cannot be empty.");
        TORCH_CHECK(coefficients.dim() == 1, "coefficients must be a 1D tensor.");
        TORCH_CHECK(static_cast<int64_t>(base_functions.size()) == coefficients.size(0),
                    "Number of base functions must match the number of coefficients.");

        torch::Tensor result = torch::zeros_like(x);

        for (size_t i = 0; i < base_functions.size(); ++i)
        {
            torch::Tensor activated_x = base_functions[i](x);
            result += coefficients[i] * activated_x;
        }

        return result;
    }

    auto LinComb::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        // return xt::activations::lin_comb(torch::zeros(10));
    }
}
