#include "include/transforms/image/affine.h"

namespace xt::transforms::image
{
    Affine::Affine() = default;

    Affine::Affine(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Affine::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
