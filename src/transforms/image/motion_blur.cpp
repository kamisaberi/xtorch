#include "include/transforms/image/motion_blur.h"

namespace xt::transforms::image
{
    MotionBlur::MotionBlur() = default;

    MotionBlur::MotionBlur(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MotionBlur::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
