#include "include/transforms/image/random_auto_contrast.h"

namespace xt::transforms::image
{
    RandomAutoContrast::RandomAutoContrast() = default;

    RandomAutoContrast::RandomAutoContrast(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomAutoContrast::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
