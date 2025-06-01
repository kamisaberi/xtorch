#include "include/transforms/image/random_horizontal_flip.h"

namespace xt::transforms::image
{
    RandomHorizontalFlip::RandomHorizontalFlip() = default;

    RandomHorizontalFlip::RandomHorizontalFlip(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomHorizontalFlip::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
