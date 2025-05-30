#include "include/dropouts/drop_path.h"

namespace xt::dropouts
{
    torch::Tensor drop_path(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DropPath::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::drop_path(torch::zeros(10));
    }
}
