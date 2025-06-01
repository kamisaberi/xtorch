#include "include/transforms/image/random_grid_dropout.h"

namespace xt::transforms::image
{
    RandomGridDropout::RandomGridDropout() = default;

    RandomGridDropout::RandomGridDropout(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomGridDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
