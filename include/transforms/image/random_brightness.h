#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomBrightness final : public xt::Module
    {
    public:
        RandomBrightness();
        explicit RandomBrightness(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
