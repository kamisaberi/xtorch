#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class PixelDropout final : public xt::Module
    {
    public:
        PixelDropout();
        explicit EdgeAdd(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
