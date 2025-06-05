#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomAutoContrast final : public xt::Module
    {
    public:
        RandomAutoContrast();
        explicit RandomAutoContrast(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
