#pragma once

#include "../common.h"


namespace xt::transforms::weather
{
    class VegetationMotion final : public xt::Module
    {
    public:
        VegetationMotion();
        explicit VegetationMotion(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
