#include "include/transforms/image/solarize.h"

namespace xt::transforms::image
{
    Solarize::Solarize() = default;

    Solarize::Solarize(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Solarize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
