#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class ZoomBlur final : public xt::Module
    {
    public:
        ZoomBlur();
        explicit ZoomBlur(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
