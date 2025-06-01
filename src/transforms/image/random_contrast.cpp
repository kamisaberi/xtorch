#include "include/transforms/image/random_contrast.h"

namespace xt::transforms::image
{
    RandomContrast::RandomContrast() = default;

    RandomContrast::RandomContrast(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomContrast::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
