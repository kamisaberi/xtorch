#include "include/transforms/image/random_rotate45.h"

namespace xt::transforms::image
{
    RandomRotate45::RandomRotate45() = default;

    RandomRotate45::RandomRotate45(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomRotate45::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
