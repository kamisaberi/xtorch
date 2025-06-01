#include "include/transforms/image/grid_shuffle.h"

namespace xt::transforms::image
{
    GridShuffle::GridShuffle() = default;

    GridShuffle::GridShuffle(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto GridShuffle::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
