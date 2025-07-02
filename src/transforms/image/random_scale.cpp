#include "include/transforms/image/random_scale.h"

namespace xt::transforms::image
{
    RandomScale::RandomScale() = default;

    RandomScale::RandomScale(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomScale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
