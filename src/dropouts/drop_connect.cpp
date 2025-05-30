#include "include/dropouts/drop_connect.h"

namespace xt::dropouts
{
    torch::Tensor drop_connect(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DropConnect::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::drop_connect(torch::zeros(10));
    }
}
