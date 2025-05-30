#include "include/activations/leaf.h"

namespace xt::activations
{
    torch::Tensor leaf(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto LEAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::leaf(torch::zeros(10));
    }
}
