#include "include/transforms/image/random_shadow.h"

namespace xt::transforms::image
{
    RandomShadow::RandomShadow() = default;

    RandomShadow::RandomShadow(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomShadow::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
