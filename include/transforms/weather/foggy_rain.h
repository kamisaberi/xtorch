#pragma once

#include "../common.h"


namespace xt::transforms::weather
{
    class FoggyRain final : public xt::Module
    {
    public:
        FoggyRain();
        explicit FoggyRain(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
