#include "include/transforms/image/sharpen.h"

namespace xt::transforms::image
{
    Sharpen::Sharpen() = default;

    Sharpen::Sharpen(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Sharpen::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
