#pragma once

#include "../common.h"


namespace xt::transforms::weather
{
    class SunFlare final : public xt::Module
    {
    public:
        SunFlare();
        explicit SunFlare(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
