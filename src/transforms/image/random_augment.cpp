#include "include/transforms/image/random_augment.h"

namespace xt::transforms::image
{
    RandomAugment::RandomAugment() = default;

    RandomAugment::RandomAugment(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomAugment::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
