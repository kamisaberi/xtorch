#include "include/transforms/image/advanced_blur.h"

namespace xt::transforms::image
{
    AdvancedBlur::AdvancedBlur() = default;

    AdvancedBlur::AdvancedBlur(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto AdvancedBlur::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
