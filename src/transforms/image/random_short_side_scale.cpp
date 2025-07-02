#include "include/transforms/image/random_short_side_scale.h"

namespace xt::transforms::image
{
    RandomShortSideScale::RandomShortSideScale() = default;

    RandomShortSideScale::RandomShortSideScale(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomShortSideScale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
