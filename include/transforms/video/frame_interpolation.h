#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::video
{
    class FrameInterpolation final : public xt::Module
    {
    public:
        FrameInterpolation();
        explicit FrameInterpolation(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
