#include "include/transforms/image/random_vertical_flip.h"

namespace xt::transforms::image
{
    RandomVerticalFlip::RandomVerticalFlip() = default;

    RandomVerticalFlip::RandomVerticalFlip(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomVerticalFlip::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
