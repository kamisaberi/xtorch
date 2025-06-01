#include "include/transforms/image/random_solarize.h"

namespace xt::transforms::image
{
    RandomSolarize::RandomSolarize() = default;

    RandomSolarize::RandomSolarize(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomSolarize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
