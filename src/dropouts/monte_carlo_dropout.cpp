#include "include/dropouts/monte_carlo_dropout.h"

namespace xt::dropouts
{
    torch::Tensor monte_carlo_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto MonteCarloDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::monte_carlo_dropout(torch::zeros(10));
    }
}
