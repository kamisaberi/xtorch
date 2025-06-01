#include "include/transforms/image/random_rotate90.h"

namespace xt::transforms::image
{
    RandomRotate90::RandomRotate90() = default;

    RandomRotate90::RandomRotate90(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomRotate90::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
