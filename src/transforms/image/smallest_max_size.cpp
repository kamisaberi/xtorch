#include "include/transforms/image/smallest_max_size.h"

namespace xt::transforms::image
{
    SmallestMaxSize::SmallestMaxSize() = default;

    SmallestMaxSize::SmallestMaxSize(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto SmallestMaxSize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
