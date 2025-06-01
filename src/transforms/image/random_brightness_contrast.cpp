#include "include/transforms/image/random_brightness_contrast.h"

namespace xt::transforms::image
{
    RandomBrightnessContrast::RandomBrightnessContrast() = default;

    RandomBrightnessContrast::RandomBrightnessContrast(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomBrightnessContrast::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
