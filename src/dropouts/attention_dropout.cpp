#include "include/dropouts/attention_dropout.h"

namespace xt::dropouts
{
    torch::Tensor attention_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto AttentionDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::attention_dropout(torch::zeros(10));
    }
}
