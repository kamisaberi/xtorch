#include "include/transforms/image/black_white.h"

namespace xt::transforms::image
{
    BlackWhite::BlackWhite() = default;

    BlackWhite::BlackWhite(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto BlackWhite::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
