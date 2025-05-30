#include "include/activations/gclu.h"

namespace xt::activations
{
    torch::Tensor gclu(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GCLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::gclu(torch::zeros(10));
    }
}
