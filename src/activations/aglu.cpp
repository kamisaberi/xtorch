//TODO SHOULD IMPLEMENT
#include "include/activations/aglu.h"

namespace xt::activations
{
    torch::Tensor aglu(const torch::Tensor x, double s)
    {
        return x * torch::sigmoid(s * x);
    }

    auto AGLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::aglu(torch::zeros(10));
    }
}
