#include "include/activations/aria.h"

namespace xt::activations
{
    torch::Tensor aria(torch::Tensor x)
    {
         return  torch::zeros(10);
    }

    auto ARiA::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::aria(torch::zeros(10));
    }
}
