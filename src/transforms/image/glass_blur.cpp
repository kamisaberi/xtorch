#include "include/transforms/image/glass_blur.h"

namespace xt::transforms::image
{
    GlassBlur::GlassBlur() = default;

    GlassBlur::GlassBlur(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto GlassBlur::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
