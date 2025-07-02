#include "include/transforms/image/perspective.h"

namespace xt::transforms::image
{
    Perspective::Perspective() = default;

    Perspective::Perspective(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Perspective::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
