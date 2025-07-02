#include "include/transforms/image/random_brightness.h"

namespace xt::transforms::image
{
    RandomBrightness::RandomBrightness() = default;

    RandomBrightness::RandomBrightness(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomBrightness::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
