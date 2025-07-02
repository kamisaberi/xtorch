#include "include/transforms/image/random_invert.h"

namespace xt::transforms::image
{
    RandomInvert::RandomInvert() = default;

    RandomInvert::RandomInvert(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomInvert::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
