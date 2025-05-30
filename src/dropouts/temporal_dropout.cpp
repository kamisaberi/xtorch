#include "include/dropouts/temporal_dropout.h"

namespace xt::dropouts
{
    torch::Tensor temporal_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto TemporalDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::temporal_dropout(torch::zeros(10));
    }
}
