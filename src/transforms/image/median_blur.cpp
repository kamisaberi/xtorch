#include "include/transforms/image/median_blur.h"

namespace xt::transforms::image
{
    MedianBlur::MedianBlur() = default;

    MedianBlur::MedianBlur(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MedianBlur::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
