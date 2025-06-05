#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class Downscale final : public xt::Module
    {
    public:
        Downscale();
        explicit Downscale(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
