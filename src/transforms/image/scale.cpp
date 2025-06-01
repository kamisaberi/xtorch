#include "include/transforms/image/scale.h"

namespace xt::transforms::image
{
    Scale::Scale() = default;

    Scale::Scale(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Scale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
