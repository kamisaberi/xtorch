#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class PixelDropout final : public xt::Module
    {
    public:
        PixelDropout();
        explicit PixelDropout(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
