#include "include/transforms/image/zoom_blur.h"

namespace xt::transforms::image
{
    ZoomBlur::ZoomBlur() = default;

    ZoomBlur::ZoomBlur(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto ZoomBlur::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
