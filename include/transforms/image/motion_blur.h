#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class MotionBlur final : public xt::Module
    {
    public:
        MotionBlur();
        explicit MotionBlur(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
