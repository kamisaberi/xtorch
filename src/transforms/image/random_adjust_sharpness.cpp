#include "include/transforms/image/random_adjust_sharpness.h"

namespace xt::transforms::image
{
    RandomAdjustSharpness::RandomAdjustSharpness() = default;

    RandomAdjustSharpness::RandomAdjustSharpness(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomAdjustSharpness::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
