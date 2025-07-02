#include "include/transforms/image/longest_max_size.h"

namespace xt::transforms::image
{
    LongestMaxSize::LongestMaxSize() = default;

    LongestMaxSize::LongestMaxSize(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LongestMaxSize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
