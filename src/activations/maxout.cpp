#include "include/activations/maxout.h"

namespace xt::activations
{
    torch::Tensor maxout(torch::Tensor x)
    {
        return torch::zeros(10);
    }


    auto Maxout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::maxout(torch::zeros(10));
    }
}
