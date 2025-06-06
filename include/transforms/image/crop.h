#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class Crop final : public xt::Module
    {
    public:
        Crop();
        explicit Crop(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
