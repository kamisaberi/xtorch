#include "include/transforms/image/pixel_dropout.h"

namespace xt::transforms::image
{
    PixelDropout::PixelDropout() = default;

    PixelDropout::PixelDropout(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto PixelDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
