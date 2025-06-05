#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomContrast final : public xt::Module
    {
    public:
        RandomContrast();
        explicit RandomContrast(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
