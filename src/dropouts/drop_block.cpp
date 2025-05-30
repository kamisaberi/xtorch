#include "include/dropouts/drop_block.h"

namespace xt::dropouts
{
    torch::Tensor drop_block(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DropBlock::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::drop_block(torch::zeros(10));
    }
}
