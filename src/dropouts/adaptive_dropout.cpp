#include "include/dropouts/adaptive_dropout.h"

namespace xt::dropouts
{
    torch::Tensor adaptive_dropout(torch::Tensor x)
    {
    }

    torch::Tensor AdaptiveDropout::forward(torch::Tensor x) const
    {
        return xt::dropouts::adaptive_dropout(x);
    }
}
