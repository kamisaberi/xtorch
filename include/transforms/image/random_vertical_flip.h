#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomVerticalFlip final : public xt::Module
    {
    public:
        RandomVerticalFlip();
        explicit RandomVerticalFlip(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
