#include "include/transforms/image/to_sepia.h"

namespace xt::transforms::image
{
    ToSepia::ToSepia() = default;

    ToSepia::ToSepia(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto ToSepia::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
