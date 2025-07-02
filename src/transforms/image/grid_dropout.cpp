#include "include/transforms/image/grid_dropout.h"

namespace xt::transforms::image
{
    GridDropout::GridDropout() = default;

    GridDropout::GridDropout(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto GridDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
