#include "include/transforms/image/downscale.h"

namespace xt::transforms::image
{
    Downscale::Downscale() = default;

    Downscale::Downscale(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Downscale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
