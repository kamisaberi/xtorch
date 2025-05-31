#include "include/transforms/image/blur.h"

namespace xt::transforms::image
{
    Blur::Blur() = default;

    Blur::Blur(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Blur::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
