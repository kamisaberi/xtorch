#include "include/transforms/video/frame_interpolation.h"

namespace xt::transforms::video
{
    FrameInterpolation::FrameInterpolation() = default;

    FrameInterpolation::FrameInterpolation(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto FrameInterpolation::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
