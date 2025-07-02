#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class GlassBlur final : public xt::Module
    {
    public:
        GlassBlur();
        explicit GlassBlur(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
