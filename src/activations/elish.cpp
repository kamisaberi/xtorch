#include "include/activations/elish.h"

namespace xt::activations
{
    torch::Tensor elish(const torch::Tensor x)
    {
        torch::Tensor sig_x = torch::sigmoid(x);
        torch::Tensor positive_values = x * sig_x;
        torch::Tensor negative_values_base = torch::exp(x) - 1.0; // This is ELU(x, alpha=1.0) for x < 0
        // Alternatively: torch::Tensor negative_values_base = torch::elu(x, 1.0);
        torch::Tensor negative_values = negative_values_base * sig_x;
        torch::Tensor result = torch::where(x >= 0, positive_values, negative_values);
        return result;
    }

    auto ELiSH::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::elish(torch::zeros(10));
    }
}
