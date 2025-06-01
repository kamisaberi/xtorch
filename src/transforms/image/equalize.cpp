#include "include/transforms/image/equalize.h"

namespace xt::transforms::image
{
    Equalize::Equalize() = default;

    Equalize::Equalize(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Equalize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
