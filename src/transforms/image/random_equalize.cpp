#include "include/transforms/image/random_equalize.h"

namespace xt::transforms::image
{
    RandomEqualize::RandomEqualize() = default;

    RandomEqualize::RandomEqualize(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomEqualize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
