#include "include/activations/splash.h"

namespace xt::activations
{
    torch::Tensor splash(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SPLASH::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::splash(torch::zeros(10));
    }
}
