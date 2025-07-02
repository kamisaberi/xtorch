#include "include/transforms/image/spatter.h"

namespace xt::transforms::image
{
    Spatter::Spatter() = default;

    Spatter::Spatter(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Spatter::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
