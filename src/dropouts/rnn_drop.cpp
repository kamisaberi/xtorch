#include "include/dropouts/rnn_drop.h"

namespace xt::dropouts
{
    torch::Tensor rnn_drop(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto RnnDrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::rnn_drop(torch::zeros(10));
    }
}
