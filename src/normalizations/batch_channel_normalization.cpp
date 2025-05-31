#include "include/normalizations/batch_channel_normalization.h"

namespace xt::norm
{
    auto BatchChannelNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
