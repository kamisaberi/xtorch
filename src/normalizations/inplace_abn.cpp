#include "include/normalizations/inplace_abn.h"

namespace xt::norm
{
    auto InplaceABN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
