#include "include/transforms/image/random_posterize.h"

namespace xt::transforms::image
{
    RandomPosterize::RandomPosterize() = default;

    RandomPosterize::RandomPosterize(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomPosterize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
