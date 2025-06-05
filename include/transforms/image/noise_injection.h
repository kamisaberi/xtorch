#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class NoiseInjection final : public xt::Module
    {
    public:
        NoiseInjection();
        explicit NoiseInjection(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
