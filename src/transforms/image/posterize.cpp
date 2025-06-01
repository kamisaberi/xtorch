#include "include/transforms/image/posterize.h"

namespace xt::transforms::image
{
    Posterize::Posterize() = default;

    Posterize::Posterize(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Posterize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
