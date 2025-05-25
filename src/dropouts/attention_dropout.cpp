#include "include/dropouts/attention_dropout.h"

namespace xt::dropouts
{
    torch::Tensor attention_dropout(torch::Tensor x)
    {
    }

    torch::Tensor AttentionDropout::forward(torch::Tensor x) const
    {
        return xt::dropouts::attention_dropout(x);
    }
}
