#include "include/activations/siren.h"

namespace xt::activations
{
    torch::Tensor siren(const torch::Tensor& x, double omega_0)
    {
        return torch::sin(omega_0 * x);
    }

    auto Siren::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::siren(torch::zeros(10));
    }
}
