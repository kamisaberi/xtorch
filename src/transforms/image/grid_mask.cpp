#include "include/transforms/image/grid_mask.h"

namespace xt::transforms::image
{
    GridMask::GridMask() = default;

    GridMask::GridMask(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto GridMask::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
