#include "include/transforms/image/random_rotation.h"

namespace xt::transforms::image
{
    RandomRotation::RandomRotation() = default;

    RandomRotation::RandomRotation(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomRotation::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
