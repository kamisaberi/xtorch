#include "include/transforms/image/upscale.h"

namespace xt::transforms::image
{
    Upscale::Upscale() = default;

    Upscale::Upscale(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Upscale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
