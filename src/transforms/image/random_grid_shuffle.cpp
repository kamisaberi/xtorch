#include "include/transforms/image/random_grid_shuffle.h"

namespace xt::transforms::image
{
    RandomGridShuffle::RandomGridShuffle() = default;

    RandomGridShuffle::RandomGridShuffle(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomGridShuffle::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
