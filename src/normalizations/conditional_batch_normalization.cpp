#include "include/normalizations/conditional_batch_normalization.h"

namespace xt::norm
{
    auto ConditionalBatchNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
