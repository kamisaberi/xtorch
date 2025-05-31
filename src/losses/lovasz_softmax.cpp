#include "include/losses/lovasz_softmax.h"

namespace xt::losses
{
    torch::Tensor lovasz_softmax(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto LovaszSoftmax::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::lovasz_softmax(torch::zeros(10));
    }
}
