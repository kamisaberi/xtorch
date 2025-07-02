#include "include/transforms/image/flip.h"

namespace xt::transforms::image
{
    Flip::Flip() = default;

    Flip::Flip(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Flip::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
