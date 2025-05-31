#include "include/normalizations/decorrelated_batch_normalization.h"

namespace xt::norm
{
    auto DecorrelatedBatchNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
