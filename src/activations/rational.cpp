#include "include/activations/rational.h"

namespace xt::activations
{
    torch::Tensor rational(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto Rational::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::rational(torch::zeros(10));
    }
}
