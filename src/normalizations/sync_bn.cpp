#include "include/normalizations/sync_bn.h"

namespace xt::norm
{
    auto SyncBN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
